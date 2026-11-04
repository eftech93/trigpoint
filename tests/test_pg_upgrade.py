"""The upgrade paths inside one server.

- ALTER EXTENSION UPDATE step by step from 0.2 (the index built at 0.2's SQL level is queried and checked after every
  step; the on-disk format does not change between SQL versions, only the catalog does). Needs a test build: the
  0.2 -> 0.3 script creates the page hooks, whose symbols a release build does not export.
- A fresh 0.5 is up to date (ALTER EXTENSION UPDATE is a no-op and the catalog's version is the default).
- REINDEX INDEX CONCURRENTLY and CREATE INDEX CONCURRENTLY under a concurrent writer that inserts and deletes:
  the new index is valid, holds every surviving row inserted meanwhile and none of the deleted ones.
The pg_dump round trip and pg_upgrade 17 -> 18 need the client and server binaries: scripts/pg_upgrade_test.sh.
"""
import struct
import threading
import time

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _lit  # noqa: E402

D = 32
VERSIONS = ["0.2", "0.3", "0.4", "0.5"]


def _connect(**kw):
    try:
        c = psycopg.connect(autocommit=True, connect_timeout=3, **kw)
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


def _corpus(n, seed=0):
    rng = np.random.default_rng(seed)
    cents = rng.normal(size=(16, D)).astype(np.float32) * 3
    x = (cents[rng.integers(0, 16, n)] + rng.normal(size=(n, D))).astype(np.float32)
    q = (cents[rng.integers(0, 16, 20)] + rng.normal(size=(20, D))).astype(np.float32)
    return x, q


def _load(cur, table, x, extra_cols=""):
    n = x.shape[0]
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"CREATE TABLE {table} (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(n, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(n); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy(f"COPY {table} (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))


def _top(cur, table, q, k=10, exact=True):
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SET trigpoint.exact = {'on' if exact else 'off'}")
    out = []
    for qv in q:
        cur.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT {k}", (_lit(qv),), prepare=False)
        out.append([r[0] for r in cur.fetchall()])
    cur.execute("RESET trigpoint.exact"); cur.execute("RESET enable_seqscan")
    return out


def _brute(x, q, k=10):
    d2 = (q ** 2).sum(1)[:, None] - 2 * q @ x.T + (x ** 2).sum(1)[None, :]
    return np.argsort(d2, axis=1)[:, :k].tolist()


def _check_ok(cur, index):
    cur.execute(f"SELECT trigpoint_check('{index}', true)")
    rows = [r[0] for r in cur.fetchall()]
    assert rows[-1].startswith("summary: ok"), rows[-5:]


@pytest.mark.hooks
def test_update_step_by_step_from_0_2(conn):
    """Install 0.2 in its own database, build an index there, then UPDATE TO 0.3, 0.4, 0.5 one step at a time; the
    index built at 0.2 answers exactly after every step and the checker (from 0.3) finds nothing."""
    conn.execute("DROP DATABASE IF EXISTS tp_upgrade_db")
    conn.execute("CREATE DATABASE tp_upgrade_db")
    c = _connect(dbname="tp_upgrade_db")
    try:
        cur = c.cursor()
        cur.execute("CREATE EXTENSION vector")
        cur.execute("CREATE EXTENSION trigpoint VERSION '0.2'")
        x, q = _corpus(6000)
        gt = _brute(x, q)
        _load(cur, "tp_up", x)
        cur.execute("CREATE INDEX tp_up_ix ON tp_up USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 200, seed = 1)")
        base = _top(cur, "tp_up", q)
        assert base == gt
        for v in VERSIONS[1:]:
            cur.execute(f"ALTER EXTENSION trigpoint UPDATE TO '{v}'")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'trigpoint'")
            assert cur.fetchone()[0] == v
            assert _top(cur, "tp_up", q) == gt, v
            _check_ok(cur, "tp_up_ix")
            # the catalog of each version: the checker from 0.3, the cosine class and the totals from 0.4, the
            # availability from 0.5; the page hooks exist from 0.3 to 0.4 and are dropped by 0.5
            have = {r[0] for r in cur.execute("SELECT proname FROM pg_proc WHERE proname LIKE 'trigpoint%'").fetchall()}
            assert "trigpoint_check" in have
            assert ("trigpoint_shared_total" in have) == (v >= "0.4")
            assert ("trigpoint_shared_available" in have) == (v >= "0.5")
            assert ("trigpoint_debug_peek" in have) == (v < "0.5")
            if v >= "0.4":
                cur.execute("SELECT count(*) FROM pg_opclass WHERE opcname = 'vector_cosine_ops' AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'trigpoint')")
                assert cur.fetchone()[0] == 1
        # a write and a VACUUM at 0.5 on the index built under 0.2's catalog
        cur.execute("DELETE FROM tp_up WHERE id % 10 = 0")
        cur.execute("VACUUM tp_up")
        _check_ok(cur, "tp_up_ix")
        live = np.array([i for i in range(x.shape[0]) if i % 10 != 0])
        assert _top(cur, "tp_up", q) == [[int(live[j]) for j in row] for row in _brute(x[live], q)]
    finally:
        c.close()
        conn.execute("DROP DATABASE IF EXISTS tp_upgrade_db")


def test_fresh_install_is_current(conn):
    cur = conn.cursor()
    cur.execute("ALTER EXTENSION trigpoint UPDATE")
    cur.execute("SELECT extversion, (SELECT default_version FROM pg_available_extensions WHERE name = 'trigpoint') FROM pg_extension WHERE extname = 'trigpoint'")
    have, want = cur.fetchone()
    assert have == want == VERSIONS[-1], (have, want)
    cur.execute("SELECT trigpoint_shared_available() IS NOT NULL")
    assert cur.fetchone()[0]


def _writer(table, base, new, deleted_ids, stop, inserted):
    with psycopg.connect(autocommit=True) as other:
        oc = other.cursor(); i = 0
        while not stop.is_set() and i < len(new):
            oc.execute(f"INSERT INTO {table} VALUES (%s, %s::vector)", (base + i, _lit(new[i])), prepare=False)
            inserted.append(base + i)
            if i % 3 == 0:
                oc.execute(f"DELETE FROM {table} WHERE id = %s", (deleted_ids[i // 3],), prepare=False)
            i += 1
            time.sleep(0.002)


@pytest.mark.parametrize("how", ["reindex", "create"])
def test_concurrently_under_a_writer(conn, how):
    """REINDEX INDEX CONCURRENTLY (an index exists) and CREATE INDEX CONCURRENTLY (none yet) while another backend
    inserts and deletes: the result is valid, the rows inserted meanwhile are found (exact mode, top-1 of their own
    vector), the deleted ones are not, and the checker is clean."""
    cur = conn.cursor()
    x, q = _corpus(8000, seed=2)
    table = f"tp_conc_{how}"
    _load(cur, table, x)
    if how == "reindex":
        cur.execute(f"CREATE INDEX {table}_ix ON {table} USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 200)")
    base = x.shape[0]
    rng = np.random.default_rng(5)
    new = (x[rng.choice(base, 300, replace=False)] + 0.01 * rng.normal(size=(300, D))).astype(np.float32)
    deleted_ids = rng.choice(base, 100, replace=False).tolist()
    inserted, stop = [], threading.Event()
    th = threading.Thread(target=_writer, args=(table, base, new, deleted_ids, stop, inserted)); th.start()
    time.sleep(0.1)
    if how == "reindex":
        cur.execute(f"REINDEX INDEX CONCURRENTLY {table}_ix")
    else:
        cur.execute(f"CREATE INDEX CONCURRENTLY {table}_ix ON {table} USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 200)")
    stop.set(); th.join()
    cur.execute(f"SELECT indisvalid, indisready FROM pg_index WHERE indexrelid = '{table}_ix'::regclass")
    assert cur.fetchone() == (True, True)
    assert len(inserted) >= 5, len(inserted)
    _check_ok(cur, f"{table}_ix")
    cur.execute("SET trigpoint.exact = on"); cur.execute("SET enable_seqscan = off")
    missing = []
    for rid in inserted:
        cur.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT 1", (_lit(new[rid - base]),), prepare=False)
        if cur.fetchone()[0] != rid:
            missing.append(rid)
    gone = deleted_ids[: (len(inserted) + 2) // 3]
    seen = set()
    for did in gone:
        cur.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT 3", (_lit(x[did]),), prepare=False)
        seen |= {r[0] for r in cur.fetchall()}
    cur.execute("RESET trigpoint.exact"); cur.execute("RESET enable_seqscan")
    assert not missing, f"{len(missing)} of {len(inserted)} rows inserted during {how} CONCURRENTLY not found: {missing[:10]}"
    assert not (seen & set(gone)), seen & set(gone)
    cur.execute(f"SELECT count(*) FROM {table}")
    assert cur.fetchone()[0] == base + len(inserted) - len(gone)
    cur.execute(f"DROP TABLE {table}")
