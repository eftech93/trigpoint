"""Hardening H11: the cosine operator class. Rows with very different norms, queries ordered by <=>; the index's
order must be the brute-force cosine order at exact = on, approximate recall must be high, inserts, deletes,
a compaction and REINDEX must keep it, and the planner must use the index for ORDER BY v <=> q.
"""

import os
import struct

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _lit  # noqa: E402

D = 48


def _connect():
    try:
        c = psycopg.connect(autocommit=True, connect_timeout=3)
        c.execute("SET statement_timeout = '%ds'" % int(300 * (5 if 'sanitizer build' in c.execute('SELECT version()').fetchone()[0] else 1) * float(os.environ.get('TP_TEST_SLOW', '1'))))
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
    c.execute("SELECT count(*) FROM pg_opclass WHERE opcname = 'vector_cosine_ops' AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'trigpoint')")
    if c.execute("SELECT count(*) FROM pg_opclass WHERE opcname = 'vector_cosine_ops' AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'trigpoint')").fetchone()[0] == 0:
        pytest.skip("trigpoint < 0.4: no cosine class")
    yield c
    c.close()


def corpus(n, seed):
    rng = np.random.default_rng(seed)
    cents = rng.normal(size=(24, D)).astype(np.float32) * 2
    x = (cents[rng.integers(0, 24, n)] + 0.7 * rng.normal(size=(n, D))).astype(np.float32)
    scale = np.exp(rng.uniform(-3, 3, size=n)).astype(np.float32)   # norms over three decades
    return x * scale[:, None]


def cosine_top(x, q, k=10):
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    d = 1 - xn @ qn
    return np.argsort(d)[:k].tolist(), np.sort(d)[:k]


def knn(cur, table, q, k=10, exact=False):
    cur.execute(f"SET trigpoint.exact = {'on' if exact else 'off'}")
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SELECT id, v <=> %s::vector FROM {table} ORDER BY 2 LIMIT {k}", (_lit(q),), prepare=False)
    out = cur.fetchall()
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET trigpoint.exact = off")
    return [r[0] for r in out], [r[1] for r in out]


def test_cosine_class_orders_like_brute_force(conn):
    cur = conn.cursor()
    x = corpus(8000, 0)
    cur.execute("DROP TABLE IF EXISTS tp_cos")
    cur.execute(f"CREATE TABLE tp_cos (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(len(x), dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(len(x)); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_cos (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("CREATE INDEX tp_cos_ix ON tp_cos USING trigpoint (v vector_cosine_ops) WITH (cells = 64, calibration_queries = 500)")
    cur.execute("SELECT trigpoint_info('tp_cos_ix')")
    assert "n=8000" in cur.fetchone()[0]
    cur.execute("SET enable_seqscan = off")
    cur.execute("EXPLAIN SELECT id FROM tp_cos ORDER BY v <=> %s::vector LIMIT 10", (_lit(x[0]),), prepare=False)
    plan = "\n".join(r[0] for r in cur.fetchall())
    cur.execute("SET enable_seqscan = on")
    assert "tp_cos_ix" in plan, plan
    rng = np.random.default_rng(1)
    qs = corpus(100, 5) * np.exp(rng.uniform(-2, 2, size=100)).astype(np.float32)[:, None]
    exact_hits = approx_hits = 0
    for q in qs:
        true_ids, true_d = cosine_top(x, q)
        ids, dists = knn(cur, "tp_cos", q, exact=True)
        exact_hits += len(set(ids) & set(true_ids))
        assert all(a <= b + 1e-6 for a, b in zip(dists, dists[1:])), "not in cosine order"
        assert abs(dists[-1] - true_d[-1]) < 1e-4, (dists[-1], true_d[-1])   # the k-th distance is the true one
        ids, _ = knn(cur, "tp_cos", q)
        approx_hits += len(set(ids) & set(true_ids))
    assert exact_hits >= 995, exact_hits          # exact = on: the cosine top-10 up to ties
    assert approx_hits >= 950, approx_hits
    # writes keep the metric: inserts, deletes, a compaction, the checker, REINDEX
    xi = corpus(2000, 7)
    for i, v in enumerate(xi):
        cur.execute("INSERT INTO tp_cos VALUES (%s, %s::vector)", (10000 + i, _lit(v)), prepare=False)
    allx = np.vstack([x, xi])
    ids_all = list(range(8000)) + [10000 + i for i in range(2000)]
    hits = 0
    for q in qs[:30]:
        true_ids, _ = cosine_top(allx, q)
        ids, _ = knn(cur, "tp_cos", q, exact=True)
        hits += len(set(ids) & set(ids_all[j] for j in true_ids))
    assert hits >= 298, hits
    cur.execute("DELETE FROM tp_cos WHERE id % 5 = 0")
    cur.execute("VACUUM tp_cos")
    cur.execute("SELECT trigpoint_info('tp_cos_ix')")
    assert "compactions=1" in cur.fetchone()[0]
    cur.execute("SELECT * FROM trigpoint_check('tp_cos_ix', true)")
    lines = [r[0] for r in cur.fetchall()]
    assert lines[-1].startswith("summary: ok"), lines
    keep = [j for j, i in enumerate(ids_all) if i % 5 != 0]
    hits = 0
    for q in qs[:30]:
        true_ids, _ = cosine_top(allx[keep], q)
        ids, _ = knn(cur, "tp_cos", q, exact=True)
        hits += len(set(ids) & set(ids_all[keep[j]] for j in true_ids))
    assert hits >= 298, hits
    cur.execute("REINDEX INDEX tp_cos_ix")
    ids, _ = knn(cur, "tp_cos", qs[0], exact=True)
    true_ids, _ = cosine_top(allx[keep], qs[0])
    assert set(ids) == set(ids_all[keep[j]] for j in true_ids)
    cur.execute("DROP TABLE tp_cos")


def test_cosine_index_on_an_empty_table_then_inserts(conn):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_cos0")
    cur.execute(f"CREATE TABLE tp_cos0 (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    cur.execute("CREATE INDEX tp_cos0_ix ON tp_cos0 USING trigpoint (v vector_cosine_ops)")
    x = corpus(500, 3)
    for i, v in enumerate(x):
        cur.execute("INSERT INTO tp_cos0 VALUES (%s, %s::vector)", (i, _lit(v)), prepare=False)
    q = corpus(1, 4)[0]
    true_ids, _ = cosine_top(x, q)
    ids, _ = knn(cur, "tp_cos0", q, exact=True)
    assert set(ids) == set(true_ids)
    cur.execute("DROP TABLE tp_cos0")
