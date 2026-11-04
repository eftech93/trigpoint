"""Hardening H7: edge inputs and DDL around a Trigpoint index. Each case either works or raises a clean
ERROR; the module connection must survive every case (a crash would drop it), and every index that
survives a case passes trigpoint_check(deep).
"""

import struct

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _corpus, _lit  # noqa: E402


def _connect():
    try:
        c = psycopg.connect(autocommit=True, connect_timeout=3)
        c.execute("SET statement_timeout = '%ds'" % (120 * (5 if 'sanitizer build' in c.execute('SELECT version()').fetchone()[0] else 1)))
        c.execute("SET client_min_messages = warning")
        return c
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres not reachable: {e}")


@pytest.fixture(scope="module")
def conn():
    c = _connect()
    cur = c.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"trigpoint extension not installed: {e}")
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'trigpoint'")
    if cur.fetchone()[0] < "0.3":
        pytest.skip("needs trigpoint >= 0.3 (trigpoint_check)")
    yield c
    c.close()


@pytest.fixture()
def cur(conn):
    k = conn.cursor()
    yield k
    # the connection must have survived the case
    k.execute("SELECT 1")
    assert k.fetchone()[0] == 1


def rand_rows(n, D, seed=0):
    rng = np.random.default_rng(seed)
    cents = rng.normal(size=(min(32, max(1, n)), D)).astype(np.float32) * 3
    return (cents[rng.integers(0, len(cents), n)] + rng.normal(size=(n, D))).astype(np.float32)


def fill(cur, table, x):
    n, D = x.shape
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(n, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(n); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy(f"COPY {table} (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))


def make(cur, table, x, opts="cells = 16, calibration_queries = 100", kind="TABLE"):
    n, D = x.shape
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"CREATE {kind} {table} (id int PRIMARY KEY, v vector({D}))")
    if n:
        fill(cur, table, x)
    cur.execute(f"CREATE INDEX {table}_ix ON {table} USING trigpoint (v vector_l2_ops) WITH ({opts})")


def clean(cur, index):
    cur.execute(f"SELECT * FROM trigpoint_check('{index}', true)")
    lines = [r[0] for r in cur.fetchall()]
    assert lines[-1].startswith("summary: ok"), "\n".join(lines)


def knn(cur, table, q, k=10, exact=False):
    cur.execute(f"SET trigpoint.exact = {'on' if exact else 'off'}")
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT {k}", (_lit(q),), prepare=False)
    out = [r[0] for r in cur.fetchall()]
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET trigpoint.exact = off")
    return out


def brute(x, q, k=10):
    d = ((x - q) ** 2).sum(1)
    return np.argsort(d)[:k].tolist()


# ---- rows and dimensions -----------------------------------------------------------------------------

def test_empty_table_builds_and_answers(cur):
    """H14: an index created over no rows carries the trivial model and answers every later insert exactly."""
    make(cur, "tp_e0", np.zeros((0, 8), np.float32))
    clean(cur, "tp_e0_ix")
    assert knn(cur, "tp_e0", np.zeros(8)) == []
    cur.execute("INSERT INTO tp_e0 VALUES (1, %s::vector)", (_lit(np.ones(8)),))
    assert knn(cur, "tp_e0", np.ones(8)) == [1]
    x = rand_rows(3000, 8, seed=14)
    for i, v in enumerate(x):
        cur.execute("INSERT INTO tp_e0 VALUES (%s, %s::vector)", (100 + i, _lit(v)), prepare=False)
    for i in (0, 7, 2999):
        assert knn(cur, "tp_e0", x[i], k=1) == [100 + i]
    assert set(knn(cur, "tp_e0", x[5])) == set(100 + j for j in brute(x, x[5]))
    clean(cur, "tp_e0_ix")
    cur.execute("DELETE FROM tp_e0 WHERE id % 3 = 0")
    cur.execute("VACUUM tp_e0")             # a compaction over the trivial model
    clean(cur, "tp_e0_ix")
    live = {100 + j: x[j] for j in range(3000) if (100 + j) % 3 != 0}
    ids = sorted(live)
    xl = np.stack([live[i] for i in ids])
    assert set(knn(cur, "tp_e0", x[5])) == set(ids[j] for j in brute(xl, x[5]))
    cur.execute("REINDEX INDEX tp_e0_ix")   # refits a real model over the rows
    clean(cur, "tp_e0_ix")
    assert set(knn(cur, "tp_e0", x[5], exact=True)) == set(ids[j] for j in brute(xl, x[5]))
    cur.execute("DROP TABLE tp_e0")


@pytest.mark.parametrize("n", [1, 2, 7, 33])
def test_tiny_tables(cur, n):
    x = rand_rows(n, 16, seed=n)
    make(cur, "tp_tiny", x, "cells = 4, calibration_queries = 20")
    clean(cur, "tp_tiny_ix")
    for i in range(n):
        assert knn(cur, "tp_tiny", x[i], k=1, exact=True) == [i]
    assert set(knn(cur, "tp_tiny", x[0], k=n)) == set(range(n))
    cur.execute("DROP TABLE tp_tiny")


@pytest.mark.parametrize("D", [2, 3, 17, 1536, 2000])
def test_dimensions(cur, D):
    n = 400 if D < 1000 else 300
    x = rand_rows(n, D, seed=D)
    make(cur, "tp_dim", x, "cells = 8, calibration_queries = 50")
    clean(cur, "tp_dim_ix")
    hits = sum(len(set(knn(cur, "tp_dim", x[i], exact=True)) & set(brute(x, x[i]))) for i in range(10))
    assert hits == 100, f"exact mode must match brute force at D={D}: {hits}/100"
    cur.execute("DROP TABLE tp_dim")


def test_identical_vectors(cur):
    x = np.tile(np.arange(8, dtype=np.float32), (500, 1))
    make(cur, "tp_dup", x, "cells = 4, calibration_queries = 20")
    clean(cur, "tp_dup_ix")
    out = knn(cur, "tp_dup", x[0], k=10)
    assert len(out) == 10 and len(set(out)) == 10
    cur.execute("DROP TABLE tp_dup")


def test_more_neighbours_than_rows(cur):
    x = rand_rows(12, 8)
    make(cur, "tp_k", x, "cells = 4, calibration_queries = 10")
    assert len(knn(cur, "tp_k", x[0], k=100)) == 12
    assert knn(cur, "tp_k", x[0], k=0) == []
    cur.execute("DROP TABLE tp_k")


def test_limit_beyond_k_widens_the_scan(cur):
    """H15: LIMIT larger than trigpoint.k (default 10) must return every requested row; at exact = on in
    brute-force order, and a fresh scan must not repeat rows across the widenings."""
    x = rand_rows(3000, 16, seed=15)
    make(cur, "tp_wide", x, "cells = 16, calibration_queries = 100")
    for k in (10, 11, 25, 64, 300, 1000, 3000, 5000):
        out = knn(cur, "tp_wide", x[5], k=k, exact=True)
        assert len(out) == min(k, 3000), (k, len(out))
        assert len(set(out)) == len(out), (k, "duplicates")
        assert set(out) == set(brute(x, x[5], k=min(k, 3000))), (k, "membership differs from brute force")
        # the rows come back in the index's own distance order (ties between float32 sums may differ from numpy's)
        cur.execute("SELECT v <-> %s::vector FROM tp_wide JOIN unnest(%s::int[]) WITH ORDINALITY AS u(id, o) USING (id) ORDER BY o",
                    (_lit(x[5]), out), prepare=False)
        ds = [r[0] for r in cur.fetchall()]
        assert all(a <= b * (1 + 1e-5) for a, b in zip(ds, ds[1:])), (k, "not in distance order")
    out = knn(cur, "tp_wide", x[7], k=200)   # approximate: the requested count, no duplicates, mostly right
    assert len(out) == 200 and len(set(out)) == 200
    assert len(set(out) & set(brute(x, x[7], k=200))) >= 180
    cur.execute("SET trigpoint.k = 3")
    assert knn(cur, "tp_wide", x[9], k=40, exact=True) == brute(x, x[9], k=40)
    cur.execute("RESET trigpoint.k")
    cur.execute("DROP TABLE tp_wide")


# ---- bad inputs ----------------------------------------------------------------------------------------

def test_null_vectors_are_skipped_by_the_index(cur):
    x = rand_rows(200, 8)
    make(cur, "tp_null", x)
    cur.execute("INSERT INTO tp_null VALUES (1000, NULL)")
    cur.execute("INSERT INTO tp_null VALUES (1001, %s::vector)", (_lit(x[0] + 0.01),))
    out = knn(cur, "tp_null", x[0], k=3)
    assert 1001 in out and 1000 not in out
    cur.execute("UPDATE tp_null SET v = NULL WHERE id = 1001")
    assert 1001 not in knn(cur, "tp_null", x[0], k=3)
    clean(cur, "tp_null_ix")
    cur.execute("DROP TABLE tp_null")


def test_dimension_mismatch_and_non_finite_are_clean_errors(cur):
    x = rand_rows(100, 8)
    make(cur, "tp_bad", x)
    with pytest.raises(psycopg.Error):
        cur.execute("INSERT INTO tp_bad VALUES (500, '[1,2,3]'::vector)")
    with pytest.raises(psycopg.Error):
        cur.execute("INSERT INTO tp_bad VALUES (501, '[NaN,0,0,0,0,0,0,0]'::vector)")
    with pytest.raises(psycopg.Error):
        cur.execute("INSERT INTO tp_bad VALUES (502, '[Infinity,0,0,0,0,0,0,0]'::vector)")
    with pytest.raises(psycopg.Error):
        cur.execute("SELECT id FROM tp_bad ORDER BY v <-> '[1,2,3]'::vector LIMIT 5")
    cur.execute("SELECT id FROM tp_bad ORDER BY v <-> NULL::vector LIMIT 5")
    assert len(cur.fetchall()) == 5  # NULL distance orders arbitrarily; it must not break the scan
    clean(cur, "tp_bad_ix")
    cur.execute("DROP TABLE tp_bad")


def test_huge_values_and_zero_vectors(cur):
    x = rand_rows(300, 8)
    x[:50] = 0.0
    x[50:60] = 1e18
    x[60:70] = -1e18
    make(cur, "tp_big", x, "cells = 4, calibration_queries = 30")
    clean(cur, "tp_big_ix")
    assert set(knn(cur, "tp_big", np.zeros(8), k=50, exact=True)) == set(range(50))
    assert knn(cur, "tp_big", np.full(8, 1e18), k=10, exact=True) and all(50 <= i < 60 for i in knn(cur, "tp_big", np.full(8, 1e18), k=10, exact=True))
    cur.execute("DROP TABLE tp_big")


# ---- options and GUCs ------------------------------------------------------------------------------------

@pytest.mark.parametrize("opts", ["cells = -3", "cells = 100000000", "levels = 0", "levels = 9", "calibration_queries = -1", "seed = 'x'"])
def test_bad_reloptions_are_rejected(cur, opts):
    x = rand_rows(100, 8)
    cur.execute("DROP TABLE IF EXISTS tp_opt")
    cur.execute("CREATE TABLE tp_opt (id int PRIMARY KEY, v vector(8)) WITH (autovacuum_enabled = off)")
    fill(cur, "tp_opt", x)
    with pytest.raises(psycopg.Error):
        cur.execute(f"CREATE INDEX tp_opt_ix ON tp_opt USING trigpoint (v vector_l2_ops) WITH ({opts})")
    cur.execute("DROP TABLE tp_opt")


@pytest.mark.parametrize("guc,val", [("trigpoint.hits", -5), ("trigpoint.probes", 0), ("trigpoint.delta", 2), ("trigpoint.delta", -1), ("trigpoint.k", 0), ("trigpoint.ef", -1)])
def test_out_of_range_gucs_are_rejected(cur, guc, val):
    with pytest.raises(psycopg.Error):
        cur.execute(f"SET {guc} = {val}")


def test_extreme_but_valid_gucs_still_answer(cur):
    x = rand_rows(500, 8)
    make(cur, "tp_guc", x)
    cur.execute("SET trigpoint.hits = 1")
    cur.execute("SET trigpoint.probes = 1")
    assert len(knn(cur, "tp_guc", x[3])) == 10
    cur.execute("SET trigpoint.hits = 100000")
    cur.execute("SET trigpoint.probes = 100000")
    assert set(knn(cur, "tp_guc", x[3])) == set(brute(x, x[3]))
    cur.execute("RESET trigpoint.hits")
    cur.execute("RESET trigpoint.probes")
    cur.execute("DROP TABLE tp_guc")


# ---- DDL --------------------------------------------------------------------------------------------------

def test_reindex_truncate_drop_column(cur):
    x = rand_rows(2000, 16)
    make(cur, "tp_ddl", x)
    cur.execute("SELECT count(*) FROM trigpoint_shared() WHERE indexrelid = 'tp_ddl_ix'::regclass")
    knn(cur, "tp_ddl", x[0])  # loads the shared image
    cur.execute("REINDEX INDEX tp_ddl_ix")
    clean(cur, "tp_ddl_ix")
    assert set(knn(cur, "tp_ddl", x[1], exact=True)) == set(brute(x, x[1]))
    cur.execute("TRUNCATE tp_ddl")          # rebuilds the index over the empty heap: the trivial model (H14)
    clean(cur, "tp_ddl_ix")
    assert knn(cur, "tp_ddl", x[1]) == []
    fill(cur, "tp_ddl", x[:100])
    assert set(knn(cur, "tp_ddl", x[1], exact=True)) == set(brute(x[:100], x[1]))
    assert set(knn(cur, "tp_ddl", x[1])) == set(brute(x[:100], x[1]))   # one cell, exact bound: approximate = exact
    clean(cur, "tp_ddl_ix")
    cur.execute("ALTER TABLE tp_ddl DROP COLUMN v")
    cur.execute("SELECT count(*) FROM pg_class WHERE relname = 'tp_ddl_ix'")
    assert cur.fetchone()[0] == 0
    cur.execute("DROP TABLE tp_ddl")


@pytest.mark.shared_images
def test_drop_index_releases_the_shared_image(cur):
    x = rand_rows(3000, 16)
    make(cur, "tp_seg", x)
    knn(cur, "tp_seg", x[0])
    cur.execute("SELECT relfilenode FROM pg_class WHERE relname = 'tp_seg_ix'")
    rfn = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM trigpoint_shared() WHERE relfilenode = %s", (rfn,))
    assert cur.fetchone()[0] == 1
    cur.execute("DROP INDEX tp_seg_ix")
    # another load (any index) sweeps the orphans; the dropped relfilenode must be gone afterwards
    make(cur, "tp_seg2", x[:500])
    knn(cur, "tp_seg2", x[0])
    cur.execute("SELECT count(*) FROM trigpoint_shared() WHERE relfilenode = %s", (rfn,))
    assert cur.fetchone()[0] == 0, "the dropped index's segment is still pinned"
    cur.execute("DROP TABLE tp_seg, tp_seg2")


def test_temp_and_partitioned_tables(cur):
    x = rand_rows(500, 8)
    make(cur, "tp_tmp", x, kind="TEMP TABLE")
    clean(cur, "tp_tmp_ix")
    assert set(knn(cur, "tp_tmp", x[2], exact=True)) == set(brute(x, x[2]))
    cur.execute("DROP TABLE tp_tmp")
    cur.execute("DROP TABLE IF EXISTS tp_part")
    cur.execute("CREATE TABLE tp_part (id int, v vector(8)) PARTITION BY RANGE (id)")
    cur.execute("CREATE TABLE tp_part_a PARTITION OF tp_part FOR VALUES FROM (0) TO (250)")
    cur.execute("CREATE TABLE tp_part_b PARTITION OF tp_part FOR VALUES FROM (250) TO (1000)")
    fill(cur, "tp_part", x)
    cur.execute("CREATE INDEX tp_part_ix ON tp_part USING trigpoint (v vector_l2_ops) WITH (cells = 4, calibration_queries = 20)")
    clean(cur, "tp_part_a_v_idx")
    clean(cur, "tp_part_b_v_idx")
    assert set(knn(cur, "tp_part", x[2], exact=True)) == set(brute(x, x[2]))
    cur.execute("DROP TABLE tp_part")


def test_expression_index_and_where_filter(cur):
    x = rand_rows(800, 8)
    make(cur, "tp_expr", x)
    cur.execute("SET enable_seqscan = off")
    cur.execute("SELECT id FROM tp_expr WHERE id %% 2 = 0 ORDER BY v <-> %s::vector LIMIT 5", (_lit(x[4]),), prepare=False)
    out = [r[0] for r in cur.fetchall()]
    assert all(i % 2 == 0 for i in out) and len(out) == 5
    cur.execute("SET enable_seqscan = on")
    cur.execute("DROP TABLE tp_expr")


def test_concurrent_build_and_vacuum_full(cur):
    x = rand_rows(2000, 8)
    cur.execute("DROP TABLE IF EXISTS tp_cic")
    cur.execute("CREATE TABLE tp_cic (id int PRIMARY KEY, v vector(8)) WITH (autovacuum_enabled = off)")
    fill(cur, "tp_cic", x)
    cur.execute("CREATE INDEX CONCURRENTLY tp_cic_ix ON tp_cic USING trigpoint (v vector_l2_ops) WITH (cells = 8, calibration_queries = 50)")
    clean(cur, "tp_cic_ix")
    cur.execute("DELETE FROM tp_cic WHERE id < 500")
    cur.execute("VACUUM FULL tp_cic")   # rewrites the heap and rebuilds every index
    clean(cur, "tp_cic_ix")
    assert set(knn(cur, "tp_cic", x[600], exact=True)) == set(i + 500 for i in brute(x[500:], x[600]))
    cur.execute("DROP TABLE tp_cic")


def test_check_on_a_table_with_toasted_vectors(cur):
    x = rand_rows(300, 1536, seed=3)   # 6 KB per vector: TOASTed on default storage
    make(cur, "tp_toast", x, "cells = 8, calibration_queries = 50")
    clean(cur, "tp_toast_ix")
    assert set(knn(cur, "tp_toast", x[7], exact=True)) == set(brute(x, x[7]))
    cur.execute("UPDATE tp_toast SET v = %s::vector WHERE id = 7", (_lit(x[8]),))
    assert knn(cur, "tp_toast", x[8], k=2, exact=True) == [8, 7] or knn(cur, "tp_toast", x[8], k=2, exact=True) == [7, 8]
    clean(cur, "tp_toast_ix")
    cur.execute("DROP TABLE tp_toast")


# ---- resource governance (H8) and the replay bound (H9) -------------------------------------------------

@pytest.mark.shared_images
def test_shared_memory_cap_refuses_a_load_and_the_totals_report_it(cur):
    x = rand_rows(6000, 64, seed=8)
    make(cur, "tp_cap", x, "cells = 16, calibration_queries = 100")
    cur.execute("SELECT (trigpoint_shared_total()).*")
    images0, loading0, bytes0, max0 = cur.fetchone()
    assert max0 == 0
    cur.execute("SET trigpoint.max_shared_mb = 1")   # a 6k x 64-d image is ~2 MB
    try:
        with pytest.raises(psycopg.Error, match="max_shared_mb"):
            knn(cur, "tp_cap", x[0])
        cur.execute("SET trigpoint.shared_image = off")   # the owned path does not count against the cap
        assert set(knn(cur, "tp_cap", x[0], exact=True)) == set(brute(x, x[0]))
        cur.execute("RESET trigpoint.shared_image")
    finally:
        cur.execute("RESET trigpoint.max_shared_mb")
    assert set(knn(cur, "tp_cap", x[0], exact=True)) == set(brute(x, x[0]))
    cur.execute("SELECT (trigpoint_shared_total()).*")
    images1, loading1, bytes1, max1 = cur.fetchone()
    cur.execute("SELECT count(*), coalesce(sum(bytes), 0) FROM trigpoint_shared() WHERE indexrelid = 'tp_cap_ix'::regclass")
    n_mine, bytes_mine = cur.fetchone()
    # the sweep inside the cap check may have dropped stale entries of earlier tests, so compare against this index only
    assert n_mine == 1 and bytes_mine > 0 and bytes1 >= bytes_mine and loading1 == 0 and max1 == 0 and images1 >= 1
    cur.execute("DROP TABLE tp_cap")


def test_compaction_is_triggered_by_the_record_count(cur):
    x = rand_rows(4000, 16, seed=9)
    make(cur, "tp_rec", x, "cells = 16, calibration_queries = 100")
    cur.execute("SET trigpoint.compact_frac = 100")      # never by share
    cur.execute("SET trigpoint.compact_records = 1000")
    try:
        for i in range(600):
            cur.execute("INSERT INTO tp_rec VALUES (%s, %s::vector)", (10000 + i, _lit(x[i] + 0.01)), prepare=False)
        cur.execute("VACUUM tp_rec")
        cur.execute("SELECT trigpoint_info('tp_rec_ix')")
        assert "generation=0" in cur.fetchone()[0]          # 600 records: below the bound, no compaction
        for i in range(600):
            cur.execute("INSERT INTO tp_rec VALUES (%s, %s::vector)", (20000 + i, _lit(x[i] + 0.02)), prepare=False)
        cur.execute("VACUUM tp_rec")
        cur.execute("SELECT trigpoint_info('tp_rec_ix')")
        info = cur.fetchone()[0]
        assert "generation=1" in info and "log_inserts=0" in info, info   # 1,200 records: compacted
    finally:
        cur.execute("RESET trigpoint.compact_frac")
        cur.execute("RESET trigpoint.compact_records")
    clean(cur, "tp_rec_ix")
    allx = np.vstack([x, x[:600] + 0.01, x[:600] + 0.02])
    ids = list(range(4000)) + [10000 + i for i in range(600)] + [20000 + i for i in range(600)]
    assert set(knn(cur, "tp_rec", x[3], exact=True)) == set(ids[j] for j in brute(allx, x[3]))
    cur.execute("DROP TABLE tp_rec")
