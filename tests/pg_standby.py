"""Hardening H6: a hot standby answers from a Trigpoint index built on the primary.

The standby's backends build their shared image from the replayed pages; writes, VACUUM and compactions
happen on the primary only and reach the standby through generic WAL. Checks, after waiting for replay:

  same       the standby returns the same ids as the primary for the same queries (exact and approximate)
  check      trigpoint_check(..., true) on the standby reports 0 problems
  writes     after inserts + deletes on the primary the standby follows; after a compaction it loads the new
             generation and still answers like the primary
  readonly   an INSERT on the standby fails with the read-only error, not a crash

    PGHOST=localhost PGPORT=5433 PGUSER=vsroute PGPASSWORD=vsroute PGDATABASE=vsroute \\
        .venv/bin/python tests/pg_standby.py --standby-port 5434 --primary-container trigpoint-postgres-local
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import time

import numpy as np
import psycopg

D = 32


def lit(v):
    return "[" + ",".join(repr(float(t)) for t in v) + "]"


def connect(port=None, retries=90):
    last = None
    for _ in range(retries):
        try:
            c = psycopg.connect(autocommit=True, connect_timeout=3, port=port) if port else psycopg.connect(autocommit=True, connect_timeout=3)
            c.execute("SET statement_timeout = '120s'")
            c.execute("SET client_min_messages = warning")
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"not reachable: {last}")


def wait_replay(prim, stby, timeout=120):
    lsn = prim.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = stby.execute("SELECT pg_last_wal_replay_lsn() >= %s::pg_lsn", (lsn,)).fetchone()[0]
        if r:
            return True
        time.sleep(0.2)
    return False


def knn(c, table, q, k=10, exact=False):
    c.execute(f"SET trigpoint.exact = {'on' if exact else 'off'}")
    c.execute("SET enable_seqscan = off")
    out = [r[0] for r in c.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT {k}", (lit(q),), prepare=False).fetchall()]
    c.execute("SET enable_seqscan = on")
    c.execute("SET trigpoint.exact = off")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standby-port", type=int, default=5434)
    ap.add_argument("--primary-container", default="trigpoint-postgres-local")
    ap.add_argument("--compose", default="docker/compose.local.yml")
    args = ap.parse_args()
    fails = []

    def expect(cond, what):
        print(("  ok   " if cond else "  FAIL ") + what, flush=True)
        if not cond:
            fails.append(what)

    prim = connect()
    prim.execute("CREATE EXTENSION IF NOT EXISTS vector")
    prim.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    # replication access for the standby's base backup and streaming (idempotent)
    subprocess.run(["docker", "exec", args.primary_container, "sh", "-c",
                    "grep -q 'host replication all all' $PGDATA/pg_hba.conf || echo 'host replication all all md5' >> $PGDATA/pg_hba.conf; "
                    "grep -q '^host all all 0.0.0.0/0 md5' $PGDATA/pg_hba.conf || true"], check=True)
    prim.execute("SELECT pg_reload_conf()")
    subprocess.run(["docker", "compose", "-f", args.compose, "--profile", "standby", "up", "-d", "standby"], check=True, capture_output=True)
    stby = connect(port=args.standby_port)
    expect(stby.execute("SELECT pg_is_in_recovery()").fetchone()[0], "standby is in recovery")
    sid_p = prim.execute("SELECT system_identifier FROM pg_control_system()").fetchone()[0]
    sid_s = stby.execute("SELECT system_identifier FROM pg_control_system()").fetchone()[0]
    if sid_p != sid_s:
        print(f"  FAIL standby belongs to system {sid_s}, the primary is {sid_p}: stale standby volume (docker compose --profile standby down -v)")
        sys.exit(1)
    expect(True, "standby is a clone of this primary")

    rng = np.random.default_rng(0)
    cents = rng.normal(size=(32, D)).astype(np.float32) * 3
    x = (cents[rng.integers(0, 32, 8000)] + rng.normal(size=(8000, D))).astype(np.float32)
    prim.execute("DROP TABLE IF EXISTS tp_stb")
    prim.execute(f"CREATE TABLE tp_stb (id int PRIMARY KEY, v vector({D}))")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(len(x), dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(len(x)); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    cur = prim.cursor()
    with cur.copy("COPY tp_stb (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    prim.execute("CREATE INDEX tp_stb_ix ON tp_stb USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    expect(wait_replay(prim, stby), "the build replayed on the standby")
    qs = x[:40] + 0.3 * rng.normal(size=(40, D)).astype(np.float32)
    same = sum(knn(prim, "tp_stb", q, exact=True) == knn(stby, "tp_stb", q, exact=True) for q in qs)
    expect(same == 40, f"exact results identical on primary and standby ({same}/40)")
    same = sum(len(set(knn(prim, "tp_stb", q)) & set(knn(stby, "tp_stb", q))) >= 9 for q in qs)
    expect(same == 40, f"approximate results agree on primary and standby ({same}/40 within one row)")
    lines = [r[0] for r in stby.execute("SELECT * FROM trigpoint_check('tp_stb_ix', true)").fetchall()]
    expect(lines[-1].startswith("summary: ok"), "checker clean on the standby: " + lines[-1])
    row = stby.execute("SELECT count(*) FROM trigpoint_shared() WHERE indexrelid = 'tp_stb_ix'::regclass").fetchone()[0]
    expect(row == 1, "the standby holds its own shared image of the index")
    # writes on the primary reach the standby through the log pages
    for i in range(1500):
        prim.execute("INSERT INTO tp_stb VALUES (%s, %s::vector)", (100000 + i, lit(x[i] + 0.05)), prepare=False)
    prim.execute("DELETE FROM tp_stb WHERE id % 7 = 0")
    expect(wait_replay(prim, stby), "the inserts and deletes replayed")
    same = sum(knn(prim, "tp_stb", q, exact=True) == knn(stby, "tp_stb", q, exact=True) for q in qs)
    expect(same == 40, f"exact results identical after writes ({same}/40)")
    gen0 = stby.execute("SELECT trigpoint_info('tp_stb_ix')").fetchone()[0]
    prim.execute("VACUUM tp_stb")   # tombstones + a compaction (deletes > 20 %)
    expect(wait_replay(prim, stby), "VACUUM and the compaction replayed")
    info_p = prim.execute("SELECT trigpoint_info('tp_stb_ix')").fetchone()[0]
    info_s = stby.execute("SELECT trigpoint_info('tp_stb_ix')").fetchone()[0]
    gp = next(t for t in info_p.split() if t.startswith("generation=")); gs = next(t for t in info_s.split() if t.startswith("generation="))
    expect(gp == gs and gp != next(t for t in gen0.split() if t.startswith("generation=")), f"the standby reads the new generation ({gp} on both)")
    same = sum(knn(prim, "tp_stb", q, exact=True) == knn(stby, "tp_stb", q, exact=True) for q in qs)
    expect(same == 40, f"exact results identical after the compaction ({same}/40)")
    lines = [r[0] for r in stby.execute("SELECT * FROM trigpoint_check('tp_stb_ix', true)").fetchall()]
    expect(lines[-1].startswith("summary: ok"), "checker clean on the standby after the compaction: " + lines[-1])
    try:
        stby.execute("INSERT INTO tp_stb VALUES (999999, %s::vector)", (lit(x[0]),))
        expect(False, "an insert on the standby must fail")
    except psycopg.errors.ReadOnlySqlTransaction:
        expect(True, "an insert on the standby fails as read-only")
    except psycopg.Error as e:
        expect("read-only" in str(e), f"an insert on the standby fails cleanly: {str(e)[:60]}")
    expect(stby.execute("SELECT 1").fetchone()[0] == 1, "standby still answers")
    prim.execute("DROP TABLE tp_stb")
    print(f"\nstandby: {len(fails)} failures; {'PASS' if not fails else 'FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
