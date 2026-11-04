"""Hardening H5: concurrency soak for the Trigpoint access method.

Writers insert, delete and update rows while readers query, a VACUUM loop forces compactions and a checker
runs trigpoint_check(deep) under a SHARE lock. Every invariant is checked inside one REPEATABLE READ
snapshot, so concurrent writes cannot fake a violation:

  exact      trigpoint.exact = on returns the same distance multiset as a sequential scan (the certificate)
  no-ghost   every id an approximate scan returns exists in the snapshot (no dead or foreign TID)
  recall     approximate recall@10 against the snapshot's brute force, averaged over a window, >= --min-recall
  check      trigpoint_check(..., true) reports 0 problems at every run and at the end
  bounded    the index relation does not grow past --max-growth times its size after the first compaction

Runs against PGHOST etc. (the local loop: PGPORT=5433). Exit code 0 only when no invariant failed and no
statement errored. Statement timeouts turn a hang into a failure. Intended runs: 15 min on the Mac image
(plain and SANITIZE=1), 60 min on the server image.

    PGHOST=localhost PGPORT=5433 PGUSER=vsroute PGPASSWORD=vsroute PGDATABASE=vsroute \\
        .venv/bin/python tests/pg_soak.py --seconds 900 --writers 4 --readers 8
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import threading
import time

import numpy as np
import psycopg

D = 64
K = 10


def lit(v):
    return "[" + ",".join(repr(float(t)) for t in v) + "]"


class State:
    def __init__(self, x, cents, drift):
        self.lock = threading.RLock()   # re-entrant: violate()/bump() are called under it
        self.live = {i: x[i] for i in range(len(x))}      # id -> vector as last written (for query generation only)
        self.next_id = len(x)
        self.target = len(x)
        # new rows come from the corpus's own cluster centres (the index is measured, not the data); with --drift
        # they cluster around 32 existing rows instead, which moves the geometry the calibration was fitted on
        # (found 2026-10-28: recall 0.98 -> 0.91 under drift, 0.96 with the bound almost exact, 1.00 at exact = on)
        self.cents = x[np.random.default_rng(0).choice(len(x), 32, replace=False)] if drift else cents
        self.stop = threading.Event()
        self.errors = []
        self.violations = []
        self.counts = {"insert": 0, "delete": 0, "update": 0, "query": 0, "filtered": 0, "exact": 0, "vacuum": 0, "check": 0, "reindex": 0, "deadlock": 0}
        self.recalls = []          # the sliding window (the progress line)
        self.all_recalls = []      # every approximate query (the floor is checked on the whole run and per window at the end)

    def note_error(self, who, e):
        with self.lock:
            self.errors.append(f"{who}: {e!r}"[:300])

    def violate(self, who, what):
        with self.lock:
            self.violations.append(f"{who}: {what}"[:400])

    def bump(self, key, n=1):
        with self.lock:
            self.counts[key] += n

    def random_vec(self, rng):
        c = self.cents[rng.integers(0, len(self.cents))]
        return (c + rng.normal(size=D)).astype(np.float32)


def connect(timeout_s=60):
    c = psycopg.connect(autocommit=True, connect_timeout=5)
    c.execute(f"SET statement_timeout = '{timeout_s}s'")
    c.execute("SET client_min_messages = warning")
    return c


def writer(st: State, seed: int, update_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    who = f"writer{seed}"
    try:
        c = connect()
        cur = c.cursor()
        while not st.stop.is_set():
            op = rng.random()
            with st.lock:
                nlive = len(st.live)
            # keep the table near its starting size so the recall floor measures churn, not growth
            # (fixed cells and probes lose recall as rows per cell grow): bias the op by the drift
            if nlive > 1.1 * st.target:
                op = 0.5 + 0.5 * op          # delete or update
            elif nlive < 0.9 * st.target:
                op = 0.5 * op                # insert
            if op >= 0.8 and update_frac <= 0:
                op = 0.6                     # no updates: delete instead
            try:
                if op < 0.5:
                    with st.lock:
                        i = st.next_id
                        st.next_id += 1
                    v = st.random_vec(rng)
                    cur.execute("INSERT INTO tp_soak VALUES (%s, %s, %s::vector)", (i, i % 16, lit(v)), prepare=False)
                    with st.lock:
                        st.live[i] = v
                    st.bump("insert")
                elif op < 0.8:
                    with st.lock:
                        if len(st.live) < 1000:
                            continue
                        i = rng.choice(list(st.live.keys()))
                        st.live.pop(i, None)
                    cur.execute("DELETE FROM tp_soak WHERE id = %s", (i,), prepare=False)
                    st.bump("delete")
                else:
                    with st.lock:
                        i = rng.choice(list(st.live.keys()))
                    v = st.random_vec(rng)
                    cur.execute("UPDATE tp_soak SET v = %s::vector WHERE id = %s", (lit(v), i), prepare=False)
                    with st.lock:
                        if i in st.live:
                            st.live[i] = v
                    st.bump("update")
            except psycopg.Error as e:
                st.note_error(who, e)
                if c.closed:
                    c = connect()
                    cur = c.cursor()
        c.close()
    except Exception as e:  # noqa: BLE001
        st.note_error(who, e)


def reader(st: State, seed: int, min_recall: float, filter_frac: float = 0.0):
    rng = np.random.default_rng(1000 + seed)
    who = f"reader{seed}"
    try:
        c = connect()
        c.autocommit = False
        cur = c.cursor()
        while not st.stop.is_set():
            q = st.random_vec(rng) + 0.3 * rng.normal(size=D).astype(np.float32)
            ql = lit(q)
            try:
                cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
                # the snapshot's brute force: distances and ids by sequential scan
                # item 5e: a share of the queries carries a filter the index cannot see (WHERE c = x, 1/16 of the rows);
                # the iterative scan has to keep the stream going; the same invariants hold against the filtered truth
                where = f"WHERE c = {int(rng.integers(0, 16))}" if rng.random() < filter_frac else ""
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute(f"SELECT id, v <-> %s::vector AS d FROM tp_soak {where} ORDER BY d LIMIT {K}", (ql,), prepare=False)
                truth = cur.fetchall()
                cur.execute("SET LOCAL enable_indexscan = on")
                cur.execute("SET LOCAL enable_seqscan = off")
                if where:
                    cur.execute("SET LOCAL trigpoint.exact = off")
                    cur.execute(f"SELECT id, v <-> %s::vector AS d FROM tp_soak {where} ORDER BY d LIMIT {K}", (ql,), prepare=False)
                    got = cur.fetchall()
                    st.bump("filtered")
                    ids = [i for i, _ in got]
                    if len(ids) != len(set(ids)):
                        st.violate(who, f"duplicate ids in a filtered result: {ids}")
                    if len(got) < len(truth):
                        st.violate(who, f"filtered scan returned {len(got)} rows, the snapshot has {len(truth)} ({where})")
                    cur.execute(f"SELECT count(*) FROM tp_soak {where} AND id = ANY(%s)", (ids,), prepare=False)
                    if cur.fetchone()[0] != len(ids):
                        st.violate(who, f"ghost or unfiltered ids in a filtered result: {ids} ({where})")
                    if len(truth) == K:
                        tids = {i for i, _ in truth}
                        kth = max(d for _, d in truth)
                        hits = sum(1 for i, d in got if i in tids or abs(d - kth) < 1e-4)
                        with st.lock:
                            st.recalls.append(hits / K)
                            st.all_recalls.append(hits / K)
                            if len(st.recalls) > 400:
                                st.recalls = st.recalls[-200:]
                elif rng.random() < 0.25:
                    cur.execute("SET LOCAL trigpoint.exact = on")
                    cur.execute(f"SELECT id, v <-> %s::vector AS d FROM tp_soak ORDER BY d LIMIT {K}", (ql,), prepare=False)
                    got = cur.fetchall()
                    st.bump("exact")
                    td = sorted(round(d, 4) for _, d in truth)
                    gd = sorted(round(d, 4) for _, d in got)
                    if td != gd:
                        st.violate(who, f"exact scan distances {gd} != sequential scan {td}")
                else:
                    cur.execute("SET LOCAL trigpoint.exact = off")
                    cur.execute(f"SELECT id, v <-> %s::vector AS d FROM tp_soak ORDER BY d LIMIT {K}", (ql,), prepare=False)
                    got = cur.fetchall()
                    st.bump("query")
                    ids = [i for i, _ in got]
                    if len(ids) != len(set(ids)):
                        st.violate(who, f"duplicate ids in a result: {ids}")
                    cur.execute("SELECT count(*) FROM tp_soak WHERE id = ANY(%s)", (ids,), prepare=False)
                    if cur.fetchone()[0] != len(ids):
                        st.violate(who, f"ghost ids in a result: {ids}")
                    if len(truth) == K:
                        tids = {i for i, _ in truth}
                        kth = max(d for _, d in truth)
                        # a hit is an id in the true top-k or one tied with the k-th distance
                        hits = sum(1 for i, d in got if i in tids or abs(d - kth) < 1e-4)
                        with st.lock:
                            st.recalls.append(hits / K)
                            st.all_recalls.append(hits / K)
                            if len(st.recalls) > 400:
                                st.recalls = st.recalls[-200:]
                cur.execute("COMMIT")
            except psycopg.Error as e:
                st.note_error(who, e)
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                if c.closed:
                    c = connect()
                    c.autocommit = False
                    cur = c.cursor()
        c.close()
    except Exception as e:  # noqa: BLE001
        st.note_error(who, e)


def vacuumer(st: State, period: float):
    who = "vacuum"
    try:
        c = connect(timeout_s=300)
        cur = c.cursor()
        n = 0
        while not st.stop.wait(period):
            try:
                n += 1
                cur.execute("VACUUM (INDEX_CLEANUP ON) tp_soak" if n % 3 == 0 else "VACUUM tp_soak")
                st.bump("vacuum")
            except psycopg.Error as e:
                st.note_error(who, e)
                if c.closed:
                    c = connect(timeout_s=300)
                    cur = c.cursor()
        c.close()
    except Exception as e:  # noqa: BLE001
        st.note_error(who, e)


def reindexer(st: State, period: float):
    """item 5e: REINDEX INDEX CONCURRENTLY under the writers and readers, every `period` seconds"""
    who = "reindex"
    try:
        c = connect(timeout_s=600)
        cur = c.cursor()
        while not st.stop.wait(period):
            try:
                cur.execute("REINDEX INDEX CONCURRENTLY tp_soak_ix")
                cur.execute("SELECT indisvalid FROM pg_index WHERE indexrelid = 'tp_soak_ix'::regclass")
                if not cur.fetchone()[0]:
                    st.violate(who, "the index is invalid after REINDEX CONCURRENTLY")
                st.bump("reindex")
            except psycopg.errors.DeadlockDetected:
                # the checker's table SHARE lock against the swap phase: PostgreSQL's own resolution, not ours; the
                # invalid leftover index of a failed REINDEX CONCURRENTLY is dropped and the next period retries
                st.bump("deadlock")
                try:
                    cur.execute("SELECT indexrelid::regclass FROM pg_index WHERE indrelid = 'tp_soak'::regclass AND NOT indisvalid")
                    for (ix,) in cur.fetchall():
                        cur.execute(f"DROP INDEX {ix}")
                except psycopg.Error as e2:
                    st.note_error(who, e2)
            except psycopg.Error as e:
                st.note_error(who, e)
                if c.closed:
                    c = connect(timeout_s=600)
                    cur = c.cursor()
        c.close()
    except Exception as e:  # noqa: BLE001
        st.note_error(who, e)


def run_check(cur):
    cur.execute("BEGIN")
    cur.execute("LOCK TABLE tp_soak IN SHARE MODE")      # no writes while the pages are walked
    cur.execute("SELECT * FROM trigpoint_check('tp_soak_ix', true)")
    lines = [r[0] for r in cur.fetchall()]
    cur.execute("COMMIT")
    problems = int(lines[-1].split(" problems")[0].rsplit(" ", 1)[1])
    return problems, lines


def checker(st: State, period: float):
    who = "checker"
    try:
        c = connect(timeout_s=300)
        c.autocommit = False
        cur = c.cursor()
        while not st.stop.wait(period):
            try:
                problems, lines = run_check(cur)
                st.bump("check")
                if problems:
                    st.violate(who, "trigpoint_check: " + " | ".join(l for l in lines if l.startswith("problem")))
            except psycopg.errors.DeadlockDetected:
                st.bump("deadlock")   # against a REINDEX CONCURRENTLY swap (see reindexer)
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
            except psycopg.Error as e:
                st.note_error(who, e)
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
        c.close()
    except Exception as e:  # noqa: BLE001
        st.note_error(who, e)


def setup(n0: int):
    """the corpus: 32 Gaussian clusters; returns the rows, their centres and the fresh index size"""
    c = connect(timeout_s=600)
    cur = c.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    cur.execute("DROP TABLE IF EXISTS tp_soak")
    cur.execute(f"CREATE TABLE tp_soak (id int PRIMARY KEY, c int NOT NULL DEFAULT 0, v vector({D}))")   # c = id % 16: the filter column
    rng = np.random.default_rng(0)
    cents = rng.normal(size=(32, D)).astype(np.float32) * 3
    x = (cents[rng.integers(0, 32, n0)] + rng.normal(size=(n0, D))).astype(np.float32)
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(n0, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(n0); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_soak (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("UPDATE tp_soak SET c = id % 16")
    cur.execute("CREATE INDEX tp_soak_ix ON tp_soak USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    cur.execute("SELECT pg_relation_size('tp_soak_ix')")
    size0 = cur.fetchone()[0]
    c.close()
    return x, cents, size0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--readers", type=int, default=8)
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--vacuum-period", type=float, default=10.0)
    ap.add_argument("--check-period", type=float, default=15.0)
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--max-growth", type=float, default=4.0)
    ap.add_argument("--update-frac", type=float, default=0.2, help="share of writer operations that are UPDATEs (0 = inserts and deletes only)")
    ap.add_argument("--keep", action="store_true", help="leave tp_soak and its index in place for diagnosis")
    ap.add_argument("--drift", action="store_true", help="new rows cluster around existing rows (distribution drift) instead of the corpus centres")
    ap.add_argument("--filter-frac", type=float, default=0.0, help="share of reader queries with WHERE c = x (1/16 of the rows): the iterative scan under churn")
    ap.add_argument("--reindex-period", type=float, default=0.0, help="REINDEX INDEX CONCURRENTLY every this many seconds (0 = never)")
    args = ap.parse_args()

    x, cents, size0 = setup(args.rows)
    st = State(x, cents, args.drift)
    threads = [threading.Thread(target=writer, args=(st, i, args.update_frac), daemon=True) for i in range(args.writers)]
    threads += [threading.Thread(target=reader, args=(st, i, args.min_recall, args.filter_frac), daemon=True) for i in range(args.readers)]
    threads += [threading.Thread(target=vacuumer, args=(st, args.vacuum_period), daemon=True),
                threading.Thread(target=checker, args=(st, args.check_period), daemon=True)]
    if args.reindex_period > 0:
        threads.append(threading.Thread(target=reindexer, args=(st, args.reindex_period), daemon=True))
    t0 = time.time()
    for t in threads:
        t.start()
    last = t0
    while time.time() - t0 < args.seconds:
        time.sleep(1)
        if time.time() - last >= 30:
            last = time.time()
            with st.lock:
                r = float(np.mean(st.recalls[-200:])) if st.recalls else float("nan")
                print(f"[{int(last - t0):5d} s] {st.counts} recall(window)={r:.3f} live={len(st.live)} errors={len(st.errors)} violations={len(st.violations)}", flush=True)
        if len(st.violations) > 50:
            print("stopping early: more than 50 violations", flush=True)
            break
    elapsed = time.time() - t0
    st.stop.set()
    for t in threads:
        t.join(timeout=120)
    # final state
    c = connect(timeout_s=600)
    c.autocommit = False
    cur = c.cursor()
    problems, lines = run_check(cur)
    cur.execute("SELECT pg_relation_size('tp_soak_ix'), count(*) FROM tp_soak")
    size1, nrows = cur.fetchone()
    cur.execute("COMMIT")
    c.close()
    growth = (size1 / max(nrows, 1)) / (size0 / args.rows)   # bytes per row against the fresh index
    rc = np.array(st.all_recalls) if st.all_recalls else np.array([float("nan")])
    windows = [float(np.mean(rc[i:i + 500])) for i in range(0, len(rc), 500)] if len(rc) >= 500 else [float(np.mean(rc))]
    recall_ok = all(w >= args.min_recall for w in windows)
    if not recall_ok:
        st.violate("recall", f"a window of 500 approximate queries fell below {args.min_recall}: {[round(w, 3) for w in windows]}")
    ok = not st.violations and not st.errors and problems == 0 and growth <= args.max_growth
    print(f"\nsoak {elapsed:.0f} s: {st.counts}; rows {args.rows} -> {nrows}; index {size0 >> 10} KB -> {size1 >> 10} KB ({growth:.2f}x per row); "
          f"recall mean {float(np.mean(rc)):.3f}, windows of 500: {[round(w, 3) for w in windows]}; "
          f"final check problems={problems}; errors={len(st.errors)}; violations={len(st.violations)}; {'PASS' if ok else 'FAIL'}")
    for e in st.errors[:20]:
        print("error:", e)
    for v in st.violations[:20]:
        print("violation:", v)
    if problems:
        print("\n".join(l for l in lines if l.startswith("problem")))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
