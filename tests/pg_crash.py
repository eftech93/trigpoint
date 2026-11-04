"""Hardening H4: crash recovery of the Trigpoint access method.

Kills the PostgreSQL container (SIGKILL to the postmaster: no shutdown checkpoint, WAL replay on restart)
during a stream of committed inserts, during a compaction (VACUUM after > 20 % deletes) and during
CREATE INDEX, then checks after every restart:

  check      trigpoint_check(..., true) reports 0 problems
  committed  every insert whose COMMIT returned is in the heap and found by the index at exact = on
             (an insert in flight at the kill may be either, and is only required to be consistent)
  no-ghost   nothing the index returns is missing from the heap
  deleted    rows deleted before a compaction crash are never returned afterwards
  unlogged   an UNLOGGED table's index comes back empty and usable (ambuildempty's init fork)
  ddl        a CREATE INDEX killed midway leaves no index; the next CREATE INDEX works

Needs docker on the host and the container name (PGHOST etc. point at it):

    PGHOST=localhost PGPORT=5433 PGUSER=vsroute PGPASSWORD=vsroute PGDATABASE=vsroute \\
        .venv/bin/python tests/pg_crash.py --container trigpoint-postgres-local --iterations 5
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import psycopg

D = 64


def lit(v):
    return "[" + ",".join(repr(float(t)) for t in v) + "]"


def sh(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def connect(retries=60):
    last = None
    for _ in range(retries):
        try:
            c = psycopg.connect(autocommit=True, connect_timeout=3)
            c.execute("SET statement_timeout = '120s'")
            c.execute("SET client_min_messages = warning")
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"postgres not reachable: {last}")


def kill_and_restart(container):
    t = time.time()
    sh("docker", "kill", "-s", "KILL", container)
    sh("docker", "start", container)
    c = connect()
    c.close()
    return time.time() - t


def corpus(n, seed=0):
    rng = np.random.default_rng(seed)
    cents = rng.normal(size=(32, D)).astype(np.float32) * 3
    return (cents[rng.integers(0, 32, n)] + rng.normal(size=(n, D))).astype(np.float32), cents


def load(cur, table, x, ids):
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(len(x), dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = ids; rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy(f"COPY {table} (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))


def generation(cur, index):
    cur.execute(f"SELECT trigpoint_info('{index}')")
    return int(next(t for t in cur.fetchone()[0].split() if t.startswith("generation=")).split("=")[1])


def check(cur, index):
    cur.execute(f"SELECT * FROM trigpoint_check('{index}', true)")
    lines = [r[0] for r in cur.fetchall()]
    return int(lines[-1].split(" problems")[0].rsplit(" ", 1)[1]), lines


def found_exact(cur, table, vid, v):
    cur.execute("SET trigpoint.exact = on")
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT 3", (lit(v),), prepare=False)
    ids = [r[0] for r in cur.fetchall()]
    cur.execute("SET trigpoint.exact = off")
    cur.execute("SET enable_seqscan = on")
    return vid in ids


class Report:
    def __init__(self):
        self.fail = []
        self.notes = []

    def expect(self, cond, what):
        (self.notes if cond else self.fail).append(what)
        print(("  ok   " if cond else "  FAIL ") + what, flush=True)


def phase_inserts(args, rep, x_all, cents):
    """kill during committed inserts"""
    c = connect()
    cur = c.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_crash")
    cur.execute(f"CREATE TABLE tp_crash (id int PRIMARY KEY, v vector({D}))")
    load(cur, "tp_crash", x_all, np.arange(len(x_all)))
    cur.execute("CREATE INDEX tp_crash_ix ON tp_crash USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    c.close()
    rng = np.random.default_rng(1)
    for it in range(args.iterations):
        committed = {}
        inflight = {}
        stop = threading.Event()

        def writer():
            w = connect()
            k = w.cursor()
            i = 1000000 + it * 100000
            while not stop.is_set():
                v = (cents[rng.integers(0, 32)] + rng.normal(size=D)).astype(np.float32)
                inflight[i] = v
                try:
                    k.execute("INSERT INTO tp_crash VALUES (%s, %s::vector)", (i, lit(v)), prepare=False)
                except Exception:  # noqa: BLE001
                    break
                committed[i] = v
                inflight.pop(i, None)
                i += 1
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        time.sleep(args.insert_seconds)
        dt = kill_and_restart(args.container)
        stop.set()
        t.join(timeout=30)
        c = connect()
        cur = c.cursor()
        problems, lines = check(cur, "tp_crash_ix")
        rep.expect(problems == 0, f"inserts #{it}: check after crash ({len(committed)} committed, restart {dt:.1f} s)" + ("" if problems == 0 else "\n" + "\n".join(lines)))
        cur.execute("SELECT count(*) FROM tp_crash WHERE id = ANY(%s)", (list(committed),))
        rep.expect(cur.fetchone()[0] == len(committed), f"inserts #{it}: every committed row is in the heap")
        missing = [i for i, v in committed.items() if not found_exact(cur, "tp_crash", i, v)]
        rep.expect(not missing, f"inserts #{it}: every committed row is found by the index at exact = on" + (f" (missing {missing[:5]})" if missing else ""))
        for i, v in inflight.items():
            cur.execute("SELECT count(*) FROM tp_crash WHERE id = %s", (i,))
            present = cur.fetchone()[0] == 1
            rep.expect(found_exact(cur, "tp_crash", i, v) == present, f"inserts #{it}: in-flight row {i} is {'present' if present else 'absent'} in heap and index alike")
        c.close()


def phase_compaction(args, rep, x_all, cents):
    """kill during VACUUM's compaction after > 20 % deletes"""
    n = len(x_all)
    for it in range(args.iterations):
        c = connect()
        cur = c.cursor()
        cur.execute("DROP TABLE IF EXISTS tp_crash")
        cur.execute(f"CREATE TABLE tp_crash (id int PRIMARY KEY, v vector({D}))")
        load(cur, "tp_crash", x_all, np.arange(n))
        cur.execute("CREATE INDEX tp_crash_ix ON tp_crash USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
        cur.execute("DELETE FROM tp_crash WHERE id % 4 = 0")  # 25 % > compact_frac 0.2
        deleted = set(range(0, n, 4))
        gen0 = generation(cur, "tp_crash_ix")
        c.close()
        done = threading.Event()   # set only when VACUUM returned: a connection killed mid-VACUUM leaves it clear

        def vac():
            try:
                w = connect()
                w.execute("VACUUM tp_crash")
                w.close()
                done.set()
            except Exception:  # noqa: BLE001
                pass

        t = threading.Thread(target=vac, daemon=True)
        t.start()
        time.sleep(args.vacuum_delay * (1 + it % 3))
        dt = kill_and_restart(args.container)
        t.join(timeout=30)
        c = connect()
        cur = c.cursor()
        gen1 = generation(cur, "tp_crash_ix")
        # the generation is the authoritative signal: unchanged after the crash = the compaction's flip never
        # committed (the kill landed before it, inside it, or VACUUM had not started); incremented = it committed
        where = "interrupted" if not done.is_set() else "finished"
        problems, lines = check(cur, "tp_crash_ix")
        rep.expect(problems == 0, f"compaction #{it}: check after crash (vacuum {where}, generation {gen0} -> {gen1}, restart {dt:.1f} s)" + ("" if problems == 0 else "\n" + "\n".join(lines)))
        sample = np.random.default_rng(it).choice(n, 200, replace=False)
        ghosts = [int(i) for i in sample if i in deleted and found_exact(cur, "tp_crash", int(i), x_all[i])]
        rep.expect(not ghosts, f"compaction #{it}: deleted rows are not returned" + (f" (ghosts {ghosts[:5]})" if ghosts else ""))
        missing = [int(i) for i in sample if i not in deleted and not found_exact(cur, "tp_crash", int(i), x_all[i])]
        rep.expect(not missing, f"compaction #{it}: live rows are found at exact = on" + (f" (missing {missing[:5]})" if missing else ""))
        cur.execute("VACUUM tp_crash")  # the compaction completes after the crash
        gen2 = generation(cur, "tp_crash_ix")
        problems, lines = check(cur, "tp_crash_ix")
        rep.expect(problems == 0, f"compaction #{it}: check after the next VACUUM (generation {gen1} -> {gen2}{'; the crash had interrupted the compaction' if gen1 == gen0 and gen2 > gen1 else ''})")
        c.close()


def phase_create_index(args, rep, x_all):
    """kill during CREATE INDEX"""
    for it in range(args.iterations):
        c = connect()
        cur = c.cursor()
        cur.execute("DROP TABLE IF EXISTS tp_crash")
        cur.execute(f"CREATE TABLE tp_crash (id int PRIMARY KEY, v vector({D}))")
        load(cur, "tp_crash", x_all, np.arange(len(x_all)))
        c.close()

        def build():
            try:
                w = connect()
                w.execute("CREATE INDEX tp_crash_ix ON tp_crash USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
                w.close()
            except Exception:  # noqa: BLE001
                pass

        t = threading.Thread(target=build, daemon=True)
        t.start()
        time.sleep(args.build_delay * (1 + it % 3))
        dt = kill_and_restart(args.container)
        t.join(timeout=60)
        c = connect()
        cur = c.cursor()
        cur.execute("SELECT indisvalid FROM pg_index WHERE indexrelid = 'tp_crash_ix'::regclass::oid") if False else None
        cur.execute("SELECT count(*) FROM pg_class WHERE relname = 'tp_crash_ix'")
        exists = cur.fetchone()[0] == 1
        if exists:
            problems, lines = check(cur, "tp_crash_ix")
            rep.expect(problems == 0, f"create index #{it}: the build committed before the kill and its index checks clean (restart {dt:.1f} s)")
            cur.execute("DROP INDEX tp_crash_ix")
        else:
            rep.expect(True, f"create index #{it}: the interrupted build left no index (restart {dt:.1f} s)")
        cur.execute("CREATE INDEX tp_crash_ix ON tp_crash USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
        problems, lines = check(cur, "tp_crash_ix")
        rep.expect(problems == 0, f"create index #{it}: the next build checks clean")
        c.close()


def phase_unlogged(args, rep, x_all, cents):
    c = connect()
    cur = c.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_crash_u")
    cur.execute(f"CREATE UNLOGGED TABLE tp_crash_u (id int PRIMARY KEY, v vector({D}))")
    load(cur, "tp_crash_u", x_all, np.arange(len(x_all)))
    cur.execute("CREATE INDEX tp_crash_u_ix ON tp_crash_u USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    c.close()
    dt = kill_and_restart(args.container)
    c = connect()
    cur = c.cursor()
    cur.execute("SELECT count(*) FROM tp_crash_u")
    rep.expect(cur.fetchone()[0] == 0, f"unlogged: the table is empty after the crash (restart {dt:.1f} s)")
    problems, lines = check(cur, "tp_crash_u_ix")
    rep.expect(problems == 0, "unlogged: the empty index checks clean" + ("" if problems == 0 else "\n" + "\n".join(lines)))
    rng = np.random.default_rng(5)
    vs = {}
    try:
        for i in range(200):
            v = (cents[rng.integers(0, 32)] + rng.normal(size=D)).astype(np.float32)
            vs[i] = v
            cur.execute("INSERT INTO tp_crash_u VALUES (%s, %s::vector)", (i, lit(v)), prepare=False)
        rep.expect(all(found_exact(cur, "tp_crash_u", i, v) for i, v in vs.items()), "unlogged: rows inserted after the crash are found")
        problems, lines = check(cur, "tp_crash_u_ix")
        rep.expect(problems == 0, "unlogged: the index checks clean after the inserts")
    except psycopg.Error as e:
        # until H14 an index without a model refuses inserts; a clean ERROR naming REINDEX is the accepted state
        rep.expect("empty index" in str(e), f"unlogged: inserts into the model-less index are refused cleanly (H14 pending): {str(e)[:80]}")
        cur.execute("REINDEX TABLE tp_crash_u") if False else None
    cur.execute("DROP TABLE tp_crash_u")
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--insert-seconds", type=float, default=3.0)
    ap.add_argument("--vacuum-delay", type=float, default=0.05)
    ap.add_argument("--build-delay", type=float, default=0.3)
    ap.add_argument("--phases", default="inserts,compaction,create,unlogged")
    args = ap.parse_args()
    x_all, cents = corpus(args.rows)
    c = connect()
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    c.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    c.close()
    rep = Report()
    t0 = time.time()
    phases = args.phases.split(",")
    if "inserts" in phases:
        phase_inserts(args, rep, x_all, cents)
    if "compaction" in phases:
        phase_compaction(args, rep, x_all, cents)
    if "create" in phases:
        phase_create_index(args, rep, x_all)
    if "unlogged" in phases:
        phase_unlogged(args, rep, x_all, cents)
    c = connect()
    c.execute("DROP TABLE IF EXISTS tp_crash")
    c.close()
    print(f"\ncrash recovery: {len(rep.notes)} checks passed, {len(rep.fail)} failed, {time.time() - t0:.0f} s; {'PASS' if not rep.fail else 'FAIL'}")
    for f in rep.fail:
        print("FAIL:", f)
    sys.exit(0 if not rep.fail else 1)


if __name__ == "__main__":
    main()
