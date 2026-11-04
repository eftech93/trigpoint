"""The shared-memory cap and a loader that dies.

- trigpoint.on_shared_cap = error (default): a load over trigpoint.max_shared_mb fails with 53400 (the H8 test in
  test_pg_edge.py); = private: the backend reads a private copy instead, answers, and publishes no entry.
- A backend terminated while it holds the load slot (the test hook trigpoint.debug_load_sleep_ms makes the load slow)
  leaves no half-published entry: trigpoint_shared_total() shows nothing loading, and the next backend loads the image
  itself at once instead of waiting on the dead loader.
- trigpoint_shared_total() agrees with trigpoint_shared(): the same count and the same bytes.
"""
import struct
import threading
import time

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _lit  # noqa: E402

N, D = 6000, 64


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


@pytest.fixture(scope="module")
def table(conn):
    rng = np.random.default_rng(21)
    x = rng.normal(size=(N, D)).astype(np.float32)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_cap2")
    cur.execute(f"CREATE TABLE tp_cap2 (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(N, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(N); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_cap2 (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("CREATE INDEX tp_cap2_ix ON tp_cap2 USING trigpoint (v vector_l2_ops) WITH (cells = 16, calibration_queries = 100)")
    yield {"x": x, "cur": cur}
    cur.execute("DROP TABLE IF EXISTS tp_cap2")


def _knn(cur, qv):
    cur.execute("SET enable_seqscan = off")
    cur.execute("SELECT id FROM tp_cap2 ORDER BY v <-> %s::vector LIMIT 10", (_lit(qv),), prepare=False)
    out = [r[0] for r in cur.fetchall()]
    cur.execute("RESET enable_seqscan")
    return out


def _entries(cur):
    cur.execute("SELECT count(*), coalesce(sum(bytes), 0) FROM trigpoint_shared() WHERE indexrelid = 'tp_cap2_ix'::regclass")
    return cur.fetchone()


@pytest.mark.shared_images
def test_private_fallback_at_the_cap(conn, table):
    x = table["x"]
    exact = set(np.argsort(((x - x[0]) ** 2).sum(1))[:10].tolist())
    with _connect() as c:   # a fresh backend: the first load of this index in it
        cur = c.cursor()
        cur.execute("REINDEX INDEX tp_cap2_ix")   # a new generation nobody has loaded
        cur.execute("SET trigpoint.max_shared_mb = 1")   # the 6k x 64 image is ~2 MB
        cur.execute("SET trigpoint.on_shared_cap = private")
        cur.execute("SET trigpoint.exact = on")
        assert set(_knn(cur, x[0])) == exact
        assert _entries(cur)[0] == 0   # a private copy publishes nothing
        cur.execute("SET trigpoint.on_shared_cap = error")
        cur.execute("RESET trigpoint.max_shared_mb")
        assert set(_knn(cur, x[0])) == exact   # the backend keeps its private copy for the session


@pytest.mark.shared_images
@pytest.mark.hooks
def test_terminated_loader_completes_or_leaves_nothing(conn, table):
    """pg_terminate_backend during a load (the sleep hook makes it slow): the load holds the meta page's share lock,
    an LWLock, which holds interrupts off, so the loader publishes its whole entry and dies after; either way the
    registry never shows the generation as loading once the loader is gone, and the next backend is served at once."""
    cur, x = table["cur"], table["x"]
    cur.execute("REINDEX INDEX tp_cap2_ix")
    slow = _connect()
    slow.execute("SET trigpoint.debug_load_sleep_ms = 4000")
    slow_pid = slow.execute("SELECT pg_backend_pid()").fetchone()[0]
    err = []

    def loader():
        try:
            _knn(slow.cursor(), x[1])
        except psycopg.Error as e:  # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=loader); th.start()
    time.sleep(0.5)
    cur.execute("SELECT images, loading FROM trigpoint_shared_total()")
    assert cur.fetchone()[1] == 1   # the slot is claimed, nothing published yet
    cur.execute("SELECT pg_terminate_backend(%s)", (slow_pid,))
    th.join(timeout=10)
    assert not th.is_alive() and err, err
    cur.execute("SELECT loading FROM trigpoint_shared_total()")
    assert cur.fetchone()[0] == 0
    n, b = _entries(cur)
    assert n <= 1 and (n == 0 or b > 0), (n, b)   # whole or absent, never half
    t = time.perf_counter()
    with _connect() as c:
        assert len(_knn(c.cursor(), x[1])) == 10
    assert time.perf_counter() - t < 2.0
    n, b = _entries(cur)
    assert n == 1 and b > 0
    slow.close()


@pytest.mark.shared_images
@pytest.mark.hooks
def test_fatal_inside_the_load_clears_the_claim(conn, table):
    """A FATAL raised while the slot is claimed (no PG_CATCH runs for FATAL): the exit callback clears the claim, so
    nothing is left loading, no waiter sleeps on a dead loader, and the next backend loads the image itself."""
    cur, x = table["cur"], table["x"]
    cur.execute("REINDEX INDEX tp_cap2_ix")
    bad = _connect()
    bad.execute("SET trigpoint.debug_load_fatal = on")
    with pytest.raises(psycopg.Error, match="test hook: FATAL"):
        _knn(bad.cursor(), x[1])
    cur.execute("SELECT loading FROM trigpoint_shared_total()")
    assert cur.fetchone()[0] == 0
    assert _entries(cur)[0] == 0
    t = time.perf_counter()
    with _connect() as c:
        assert len(_knn(c.cursor(), x[1])) == 10
    assert time.perf_counter() - t < 2.0
    n, b = _entries(cur)
    assert n == 1 and b > 0
    bad.close()


@pytest.mark.shared_images
def test_totals_agree_with_the_listing(conn, table):
    cur, x = table["cur"], table["x"]
    _knn(cur, x[2])
    cur.execute("SELECT images, loading, bytes FROM trigpoint_shared_total()")
    images, loading, total_bytes = cur.fetchone()
    cur.execute("SELECT count(*), count(*) FILTER (WHERE loading), coalesce(sum(bytes), 0) FROM trigpoint_shared()")
    n, n_loading, listed_bytes = cur.fetchone()
    assert (images, loading, total_bytes) == (n, n_loading, listed_bytes)
