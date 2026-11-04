"""Cancel and statement_timeout in every long loop.

A 200k-row build, an exact scan, a compaction VACUUM and trigpoint_check(deep) each stop within 3 s of their
statement_timeout with sqlstate 57014 (query_canceled), the backend stays usable, the index is consistent afterwards
(checker ok, no segment left loading) and the same statement completes without the timeout.
"""
import struct
import time

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _lit  # noqa: E402

N, D = 200_000, 64


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
    c.sanitized = "sanitizer build" in c.execute("SELECT version()").fetchone()[0]
    yield c
    c.close()


@pytest.fixture(scope="module")
def table(conn):
    rng = np.random.default_rng(13)
    cents = rng.normal(size=(64, D)).astype(np.float32) * 3
    x = (cents[rng.integers(0, 64, N)] + rng.normal(size=(N, D))).astype(np.float32)
    q = (cents[rng.integers(0, 64, 5)] + rng.normal(size=(5, D))).astype(np.float32)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_cancel")
    cur.execute(f"CREATE TABLE tp_cancel (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(N, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(N); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_cancel (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    yield {"q": q, "cur": cur, "slow": 5 if conn.sanitized else 1}
    cur.execute("RESET statement_timeout")
    cur.execute("DROP TABLE IF EXISTS tp_cancel")


def _expect_cancel(cur, sql, timeout_ms, within_s, params=None):
    cur.execute(f"SET statement_timeout = {timeout_ms}")
    t = time.perf_counter()
    try:
        with pytest.raises(psycopg.errors.QueryCanceled):
            cur.execute(sql, params, prepare=False)
    finally:
        took = time.perf_counter() - t
        cur.execute("RESET statement_timeout")
    assert took <= within_s, (sql[:40], took)
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1
    return took


def _check_ok(cur):
    cur.execute("SELECT trigpoint_check('tp_cancel_ix', true)")
    rows = [r[0] for r in cur.fetchall()]
    assert rows[-1].startswith("summary: ok"), rows[-5:]
    cur.execute("SELECT loading FROM trigpoint_shared_total()")
    assert cur.fetchone()[0] == 0


BUILD = "CREATE INDEX tp_cancel_ix ON tp_cancel USING trigpoint (v vector_l2_ops) WITH (calibration_queries = 500)"


def test_build_cancels(table):
    cur, slow = table["cur"], table["slow"]
    took = _expect_cancel(cur, BUILD, 2000 * slow, 2 + 3 * slow)
    cur.execute("SELECT to_regclass('tp_cancel_ix')")
    assert cur.fetchone()[0] is None
    print(f"\nbuild cancelled after {took:.2f} s")
    t = time.perf_counter()
    cur.execute(BUILD)
    print(f"build completed in {time.perf_counter() - t:.1f} s")
    _check_ok(cur)


LONG_SCAN = ("SELECT count(*) FROM generate_series(1, 4000) g, LATERAL (SELECT id FROM tp_cancel ORDER BY v <-> %s::vector LIMIT 1 OFFSET g %% 7) s")


def test_exact_scan_cancels(table):
    """One statement of 4,000 exact scans (seconds) with a 50 ms statement_timeout: cancelled within 3 s."""
    cur, q, slow = table["cur"], table["q"], table["slow"]
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET trigpoint.exact = on")
    cur.execute("SELECT id FROM tp_cancel ORDER BY v <-> %s::vector LIMIT 10", (_lit(q[0]),), prepare=False)
    cur.fetchall()   # the first query loads the image
    t = time.perf_counter()
    cur.execute("SELECT id FROM tp_cancel ORDER BY v <-> %s::vector LIMIT 10", (_lit(q[0]),), prepare=False)
    cur.fetchall()
    one = time.perf_counter() - t
    took = _expect_cancel(cur, LONG_SCAN, 50 * slow, 3 * slow, (_lit(q[1]),))
    print(f"\nexact scan {one * 1000:.1f} ms each; 4,000 of them cancelled at 50 ms after {took * 1000:.0f} ms")
    cur.execute("SET trigpoint.exact = off")
    cur.execute("RESET enable_seqscan")
    _check_ok(cur)


def test_compaction_vacuum_cancels(table):
    cur, slow = table["cur"], table["slow"]
    cur.execute("DELETE FROM tp_cancel WHERE id % 3 = 0")   # 33 %: above compact_frac, a compaction with re-calibration
    took = _expect_cancel(cur, "VACUUM tp_cancel", 300 * slow, 3 * slow)
    print(f"\ncompaction VACUUM cancelled after {took:.2f} s")
    _check_ok(cur)
    cur.execute("SELECT trigpoint_info('tp_cancel_ix')")
    assert "compactions=0" in cur.fetchone()[0]
    t = time.perf_counter()
    cur.execute("VACUUM tp_cancel")
    print(f"VACUUM completed in {time.perf_counter() - t:.1f} s")
    cur.execute("SELECT trigpoint_info('tp_cancel_ix')")
    assert "compactions=1" in cur.fetchone()[0]
    _check_ok(cur)


def test_deep_check_cancels(table):
    cur, slow = table["cur"], table["slow"]
    cur.execute("SELECT count(*) FROM trigpoint_check('tp_cancel_ix', true)")
    t = time.perf_counter()
    cur.execute("SELECT count(*) FROM trigpoint_check('tp_cancel_ix', true)")
    full = time.perf_counter() - t
    timeout_ms = max(1, int(full * 1000 / 4))
    took = _expect_cancel(cur, "SELECT count(*) FROM trigpoint_check('tp_cancel_ix', true)", timeout_ms, 3 * slow)
    print(f"\ndeep check {full * 1000:.0f} ms, cancelled at {timeout_ms} ms after {took * 1000:.0f} ms")
    _check_ok(cur)


def test_pg_cancel_backend_during_a_scan(conn, table):
    """pg_cancel_backend from another session while a scan runs (the killer keeps sending it every 100 ms until the
    statement is over, so one lands inside it): the same 57014, the backend usable after."""
    cur, q = table["cur"], table["q"]
    cur.execute("SELECT pg_backend_pid()")
    pid = cur.fetchone()[0]
    import threading
    done = threading.Event()

    def killer():
        with psycopg.connect(autocommit=True) as other:
            while not done.is_set():
                time.sleep(0.1)
                other.execute("SELECT pg_cancel_backend(%s)", (pid,))

    cur.execute("SET enable_seqscan = off"); cur.execute("SET trigpoint.exact = on")
    th = threading.Thread(target=killer); th.start()
    try:
        with pytest.raises(psycopg.errors.QueryCanceled):
            cur.execute(LONG_SCAN, (_lit(q[0]),), prepare=False)
    finally:
        done.set(); th.join()
    cur.execute("SET trigpoint.exact = off"); cur.execute("RESET enable_seqscan")
    cur.execute("SELECT 1")
    _check_ok(cur)
