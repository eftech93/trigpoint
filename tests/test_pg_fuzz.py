"""Corrupt-page fuzzing (test builds only: the page hooks).

Random single-byte mutations over every page of a trigpoint index (meta, image, log) through the page hooks. After
each: the checker runs (deep every fourth time), a fresh backend reads the pages itself (shared_image = off) and
queries. The verdict per mutation must be one of: an ERROR whose sqlstate is XX002 (index corrupted), or an answer.
Never a crash, and the module's own connection survives every case. An answer may be wrong when the byte lands in
the model's floats (the PCA rows, a centroid, a norm: the deep checker names most of these, the scan cannot see
them without a per-row checksum, which is what PostgreSQL's data checksums are for); the share of low-recall
answers must stay small (2 of 1,000 at seed 5). At the end mutations are left in place and REINDEX repairs the
index. TP_FUZZ_ITER (default 1,000) and TP_FUZZ_SEED.
"""
import os
import random
import struct
import time

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _lit  # noqa: E402

pytestmark = pytest.mark.hooks

BLCKSZ = 8192
N, D = 5000, 48


def _connect():
    try:
        c = psycopg.connect(autocommit=True, connect_timeout=3)
        c.execute("SET client_min_messages = warning")
        return c
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres not reachable: {e}")


@pytest.fixture(scope="module")
def conn():
    c = _connect()
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    try:
        c.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"trigpoint extension not installed: {e}")
    yield c
    c.close()


def _corpus():
    rng = np.random.default_rng(11)
    cents = rng.normal(size=(16, D)).astype(np.float32) * 3
    x = (cents[rng.integers(0, 16, N)] + rng.normal(size=(N, D))).astype(np.float32)
    q = (cents[rng.integers(0, 16, 8)] + rng.normal(size=(8, D))).astype(np.float32)
    d2 = (q ** 2).sum(1)[:, None] - 2 * q @ x.T + (x ** 2).sum(1)[None, :]
    return x, q, np.argsort(d2, axis=1)[:, :10]


@pytest.fixture(scope="module")
def table(conn):
    x, q, gt = _corpus()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_fuzz")
    cur.execute(f"CREATE TABLE tp_fuzz (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(N, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(N); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_fuzz (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("CREATE INDEX tp_fuzz_ix ON tp_fuzz USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 200, seed = 3)")
    # a log too: inserts and tombstones since the image
    cur.execute("INSERT INTO tp_fuzz SELECT id + 100000, v FROM tp_fuzz WHERE id < 200")
    cur.execute("DELETE FROM tp_fuzz WHERE id BETWEEN 300 AND 400")
    yield {"x": x, "q": q, "gt": gt, "cur": cur}
    cur.execute("DROP TABLE IF EXISTS tp_fuzz")


class Pager:
    def __init__(self):
        self.c = _connect()
        self.cur = self.c.cursor()
        self.cur.execute("SET trigpoint.debug = on")

    def peek(self, blk, off):
        self.cur.execute("SELECT trigpoint_debug_peek('tp_fuzz_ix', %s, %s)", (blk, off))
        return self.cur.fetchone()[0]

    def poke(self, blk, off, val):
        self.cur.execute("SELECT trigpoint_debug_poke('tp_fuzz_ix', %s, %s, %s)", (blk, off, val))

    def nblocks(self):
        self.cur.execute("SELECT pg_relation_size('tp_fuzz_ix') / 8192")
        return int(self.cur.fetchone()[0])

    def close(self):
        self.c.close()


def check(cur, deep):
    cur.execute("SELECT trigpoint_check('tp_fuzz_ix', %s)", (deep,))
    lines = [r[0] for r in cur.fetchall()]
    assert lines and lines[-1].startswith("summary:"), lines
    return int(lines[-1].split(" problems")[0].rsplit(" ", 1)[1]), lines


def fresh_query(q, gt):
    """A fresh backend reads the pages itself. ('ok', recall) | ('error', sqlstate, message); a dropped connection
    is a crash and raises."""
    c = _connect()
    try:
        cur = c.cursor()
        cur.execute("SET trigpoint.shared_image = off")
        hits = 0
        for i in range(len(q)):
            cur.execute("SELECT id FROM tp_fuzz ORDER BY v <-> %s::vector LIMIT 10", (_lit(q[i]),), prepare=False)
            hits += len(set(r[0] for r in cur.fetchall()) & set(gt[i].tolist()))
        return ("ok", hits / (10 * len(q)))
    except psycopg.errors.OperationalError as e:
        if c.closed or "closed the connection" in str(e) or "terminated abnormally" in str(e):
            raise AssertionError(f"backend crashed: {e}")
        return ("error", e.sqlstate, str(e))
    except psycopg.Error as e:
        return ("error", e.sqlstate, str(e))
    finally:
        c.close()


def test_random_byte_mutations_error_cleanly_or_answer(conn, table):
    cur, q, gt = table["cur"], table["q"], table["gt"]
    pg = Pager()
    rng = random.Random(int(os.environ.get("TP_FUZZ_SEED", "5")))
    n_iter = int(os.environ.get("TP_FUZZ_ITER", "1000"))
    nblocks = pg.nblocks()
    base = fresh_query(q, gt)
    assert base[0] == "ok" and base[1] >= 0.9, base
    flagged = errored = answered = 0
    sqlstates, low = {}, []
    t0 = time.time()
    for it in range(n_iter):
        blk = rng.randrange(nblocks)
        off = rng.randrange(BLCKSZ)
        orig = pg.peek(blk, off)
        val = (orig + rng.randrange(1, 256)) & 0xFF
        pg.poke(blk, off, val)
        try:
            problems, lines = check(cur, deep=(it % 4 == 0))
            flagged += problems > 0
            verdict = fresh_query(q, gt)
            if verdict[0] == "error":
                errored += 1
                sqlstates[verdict[1]] = sqlstates.get(verdict[1], 0) + 1
                assert verdict[1] == "XX002", (it, blk, off, verdict)
            else:
                answered += 1
                if verdict[1] < 0.8:
                    low.append((it, blk, off, verdict[1], problems))
            cur.execute("SELECT 1")   # the suite's own connection is alive
        finally:
            pg.poke(blk, off, orig)
    problems, lines = check(cur, deep=True)
    assert problems == 0, "\n".join(lines[-5:])
    print(f"\n{n_iter} mutations over {nblocks} pages in {time.time() - t0:.0f} s: checker flagged {flagged}, "
          f"fresh backend errored {errored} (sqlstates {sqlstates}), answered {answered}, low recall {len(low)}; no crash")
    assert len(low) <= max(2, n_iter // 50), low[:10]   # 2 %: the model's floats, named by the deep checker when hit
    pg.close()


def test_reindex_repairs_a_corrupted_index(conn, table):
    cur, q, gt = table["cur"], table["q"], table["gt"]
    pg = Pager()
    rng = random.Random(9)
    nblocks = pg.nblocks()
    for _ in range(40):   # left in place: the meta page's counts and a spread of image bytes
        pg.poke(rng.randrange(nblocks), rng.randrange(24, BLCKSZ - 16), rng.randrange(256))
    pg.poke(0, 24 + 8, 0xFF)   # the meta page's n
    problems, _ = check(cur, deep=True)
    assert problems > 0
    cur.execute("REINDEX INDEX tp_fuzz_ix")
    problems, lines = check(cur, deep=True)
    assert problems == 0, "\n".join(lines[-5:])
    verdict = fresh_query(q, gt)
    assert verdict[0] == "ok" and verdict[1] >= 0.9, verdict
    pg.close()
