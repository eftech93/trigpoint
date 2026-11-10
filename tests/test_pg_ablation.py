"""Ablation switches of a test build: trigpoint.debug_centroid_router and trigpoint.debug_ch.

An in-database latency ratio mixes the router, the bound and its calibration. These two test-build GUCs remove one
part each at query time, on the same index: the centroid router ranks every cell by its centroid's distance to the
query's head (no router graph, the hits setting ignored); debug_ch replaces the calibrated tail cosines by one value at
every level (1 = the exact bound over the probed cells). On a 20k clustered synthetic corpus: both switches leave the
default scan unchanged when off, the centroid router returns full results and matches the exact answer when every cell
is probed, and the exact bound over the same probed cells never returns a worse k-th distance than the calibrated one
and equals trigpoint.exact when every cell is probed.
"""
import re
import struct

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _lit  # noqa: E402

pytestmark = pytest.mark.hooks

N, D, NQ, K = 20_000, 32, 60, 10


@pytest.fixture(scope="module")
def cur():
    try:
        c = psycopg.connect(autocommit=True, connect_timeout=3)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres not reachable: {e}")
    c.execute("SET client_min_messages = warning")
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    c.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    c.execute("LOAD 'trigpoint'")
    if c.execute("SELECT count(*) FROM pg_settings WHERE name = 'trigpoint.debug_ch'").fetchone()[0] == 0:
        pytest.skip("no ablation switches in this build")
    rng = np.random.default_rng(11)
    cents = rng.normal(size=(64, D)).astype(np.float32) * 2.0
    x = (cents[rng.integers(0, 64, N)] + rng.normal(size=(N, D))).astype(np.float32)
    q = (cents[rng.integers(0, 64, NQ)] + rng.normal(size=(NQ, D))).astype(np.float32)
    cu = c.cursor()
    cu.execute("DROP TABLE IF EXISTS tp_ablation")
    cu.execute(f"CREATE TABLE tp_ablation (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(N, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(N); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cu.copy("COPY tp_ablation (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cu.execute("CREATE INDEX tp_ablation_ix ON tp_ablation USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 300, seed = 1)")
    cu.execute("ANALYZE tp_ablation")
    for s in ("enable_seqscan = off", "enable_bitmapscan = off", "trigpoint.exact = off", "trigpoint.iterative_scan = off"):
        cu.execute(f"SET {s}")
    d2 = (x ** 2).sum(1)[None, :] - 2 * q @ x.T + (q ** 2).sum(1)[:, None]
    gt = np.argsort(d2, axis=1)[:, :K]
    yield _Ctx(cu, q, gt)
    cu.execute("DROP TABLE IF EXISTS tp_ablation")
    c.close()


class _Ctx:
    """The cursor plus the queries and their exact neighbours (psycopg cursors take no attributes)."""

    def __init__(self, cu, q, gt):
        self.cu, self.q, self.gt = cu, q, gt
        self.connection = cu.connection

    def execute(self, *a, **kw):
        return self.cu.execute(*a, **kw)

    def fetchall(self):
        return self.cu.fetchall()


def _run(cur, **gucs):
    for name, val in gucs.items():
        cur.execute(f"SET trigpoint.{name} = {val}")
    out = []
    for qv in cur.q:
        cur.execute(f"SELECT id, v <-> %s::vector FROM tp_ablation ORDER BY v <-> %s::vector LIMIT {K}", (_lit(qv), _lit(qv)), prepare=False)
        out.append(cur.fetchall())
    for name in gucs:
        cur.execute(f"RESET trigpoint.{name}")
    return out


def _recall(cur, res):
    return float(np.mean([len({r[0] for r in rows} & set(g.tolist())) / K for rows, g in zip(res, cur.gt)]))


def test_off_is_the_default_scan(cur):
    base = _run(cur, probes=4)
    off = _run(cur, probes=4, debug_centroid_router="off", debug_ch=0)
    assert [[r[0] for r in a] for a in base] == [[r[0] for r in b] for b in off]


def test_centroid_router(cur):
    cur.execute("SET trigpoint.debug = on")
    cur.execute("SET client_min_messages = notice")
    notices = []
    handler = lambda d: notices.append(d.message_primary)  # noqa: E731
    cur.connection.add_notice_handler(handler)
    _run(cur, probes=4, debug_centroid_router="on")
    cur.execute("RESET trigpoint.debug")
    cur.execute("SET client_min_messages = warning")
    cur.connection.remove_notice_handler(handler)
    assert notices and all(re.search(r"\bM=0\b", n) and re.search(r"cells_hit=0\b", n) for n in notices if n.startswith("trigpoint:"))
    few = _run(cur, probes=4, debug_centroid_router="on")
    assert all(len(rows) == K for rows in few)
    every = _run(cur, probes=64, debug_centroid_router="on", debug_ch=1)
    assert _recall(cur, every) == 1.0
    graph = _recall(cur, _run(cur, probes=4))
    cent = _recall(cur, few)
    assert 0.3 < cent <= 1.0 and 0.3 < graph <= 1.0


def test_exact_bound_over_the_probed_cells(cur):
    cal = _run(cur, probes=4)
    exb = _run(cur, probes=4, debug_ch=1)
    for a, b in zip(cal, exb):
        assert len(b) == K
        assert b[-1][1] <= a[-1][1] * (1 + 1e-5) + 1e-6   # same cells, no pruning error: never a worse k-th distance
    assert _recall(cur, exb) >= _recall(cur, cal)
    full = _run(cur, probes=64, debug_ch=1)
    exact = _run(cur, exact="on")
    assert [[r[0] for r in a] for a in full] == [[r[0] for r in b] for b in exact]
    assert _recall(cur, exact) == 1.0


def test_fixed_relaxation_prunes_more_as_it_falls(cur):
    hi = _recall(cur, _run(cur, probes=8, debug_ch=1))
    lo = _recall(cur, _run(cur, probes=8, debug_ch=0.05))
    assert lo <= hi
