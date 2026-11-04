"""Trigpoint PostgreSQL extension (spike S2): correctness from inside PostgreSQL.

Runs only when the compose `postgres` service is reachable (PGHOST etc. as set in the test
container) and `CREATE EXTENSION trigpoint` works; skipped otherwise. Uses a small table built from
the dbpedia3 corpus when its base file is available (data volume), else a synthetic 6,000 x 64 one.

Checks: EXPLAIN uses the trigpoint index; recall@10 against brute force at the default GUCs is
above the floor; exact mode returns the brute-force answer on every query; after DELETE + VACUUM
the index still answers (tombstones; the executor never sees a deleted row); inserts are found by
this backend and by a fresh one (S4 log); VACUUM compacts above the threshold (new generation);
the auto-cells / levels options are honoured.
"""
import os, re, struct, time
import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")


def _connect():
    try:
        return psycopg.connect(autocommit=True, connect_timeout=3)
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
    yield c
    c.close()


def _corpus():
    import glob
    data = os.environ.get("VSROUTE_DATA", "data")
    bases = sorted(glob.glob(os.path.join(data, "dbpedia3_base_*.npy")), key=os.path.getsize)
    rng = np.random.default_rng(0)
    if bases:
        x = np.load(bases[0], mmap_mode="r")
        x = np.ascontiguousarray(x[:12000], np.float32)
        q = np.ascontiguousarray(x[rng.choice(12000, 50, replace=False)] + 0.01 * rng.normal(size=(50, x.shape[1])), np.float32)
        return x, q
    cents = rng.normal(size=(32, 64)).astype(np.float32) * 3
    x = (cents[rng.integers(0, 32, 6000)] + rng.normal(size=(6000, 64))).astype(np.float32)
    q = (cents[rng.integers(0, 32, 50)] + rng.normal(size=(50, 64))).astype(np.float32)
    return x, q


def _brute(x, q, k=10):
    d2 = (q ** 2).sum(1)[:, None] - 2 * q @ x.T + (x ** 2).sum(1)[None, :]
    return np.argsort(d2, axis=1)[:, :k]


def _lit(v):
    return "[" + ",".join(repr(float(t)) for t in v) + "]"


@pytest.fixture(scope="module")
def table(conn):
    x, q = _corpus()
    n, D = x.shape
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_test")
    cur.execute(f"CREATE TABLE tp_test (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(n, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(n); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_test (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    t = time.time()
    cur.execute("CREATE INDEX tp_test_ix ON tp_test USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    build_s = time.time() - t
    yield {"x": x, "q": q, "gt": _brute(x, q), "build_s": build_s, "cur": cur}
    cur.execute("DROP TABLE IF EXISTS tp_test")


def _search(cur, qv, settings=None, k=10):
    for key, val in (settings or {}).items():
        cur.execute(f"SET {key} = {val}")
    cur.execute(f"SELECT id FROM tp_test ORDER BY v <-> %s::vector LIMIT {k}", (_lit(qv),), prepare=False)
    return [r[0] for r in cur.fetchall()]


def test_explain_uses_trigpoint_index(table):
    cur = table["cur"]
    cur.execute("EXPLAIN SELECT id FROM tp_test ORDER BY v <-> %s::vector LIMIT 10", (_lit(table["q"][0]),), prepare=False)
    plan = "\n".join(r[0] for r in cur.fetchall())
    assert "Index Scan using tp_test_ix" in plan, plan


def test_info_reports_build(table):
    cur = table["cur"]
    cur.execute("SELECT trigpoint_info('tp_test_ix')")
    info = cur.fetchone()[0]
    assert f"n={table['x'].shape[0]}" in info and "cells=64" in info and "build_s:" in info, info


def test_recall_at_default_gucs(table):
    cur = table["cur"]
    cur.execute("RESET trigpoint.exact")
    hits = [len(set(_search(cur, qv, {"trigpoint.hits": 100, "trigpoint.probes": 8})) & set(g.tolist())) / 10 for qv, g in zip(table["q"], table["gt"])]
    assert np.mean(hits) >= 0.9, np.mean(hits)


def test_exact_mode_matches_brute_force(table):
    cur = table["cur"]
    for qv, g in zip(table["q"], table["gt"]):
        ids = _search(cur, qv, {"trigpoint.exact": "on"})
        assert sorted(ids) == sorted(g.tolist())
    cur.execute("RESET trigpoint.exact")


def test_delta_changes_bound_not_validity(table):
    cur = table["cur"]
    # a tighter delta (smaller miss probability) can only add candidates: recall is monotone non-decreasing
    r = {}
    for delta in (0.1, 0.001):
        r[delta] = np.mean([len(set(_search(cur, qv, {"trigpoint.delta": delta, "trigpoint.probes": 8})) & set(g.tolist())) / 10 for qv, g in zip(table["q"], table["gt"])])
    cur.execute("RESET trigpoint.delta")
    assert r[0.001] >= r[0.1] - 1e-9


def test_insert_is_found_by_the_index_and_by_another_backend(conn, table):
    """S4: aminsert appends to the index's log; this backend and a fresh one (its own copy of the
    index, synced from the log) must both return the new row as the nearest neighbour of itself."""
    cur = table["cur"]
    x = table["x"]
    rng = np.random.default_rng(7)
    new = (x[rng.integers(0, x.shape[0], 30)] + 0.02 * rng.normal(size=(30, x.shape[1]))).astype(np.float32)
    base = 10 ** 6
    for i, v in enumerate(new):
        cur.execute("INSERT INTO tp_test VALUES (%s, %s::vector)", (base + i, _lit(v)))
    cur.execute("SELECT trigpoint_info('tp_test_ix')")
    assert "log_inserts=30" in cur.fetchone()[0]
    for i, v in enumerate(new):
        assert _search(cur, v, {"trigpoint.exact": "on"})[0] == base + i
        assert base + i in _search(cur, v, {"trigpoint.exact": "off", "trigpoint.probes": 8})
    with psycopg.connect(autocommit=True) as other:  # a second backend: fresh copy, log applied
        oc = other.cursor()
        for i, v in enumerate(new):
            assert _search(oc, v, {"trigpoint.exact": "on"})[0] == base + i
    cur.execute("RESET trigpoint.exact")
    # the inserted rows take part in the brute-force comparison from now on
    table["x"] = np.concatenate([x, new]); table["ids"] = np.concatenate([np.arange(x.shape[0]), base + np.arange(30)])


def test_delete_vacuum_tombstones_and_compaction(conn, table):
    """DELETE + VACUUM tombstones the rows in the log; with a low compaction threshold the next VACUUM
    rewrites the image (generation + 1) and other backends reload it; exact mode stays exact.
    Needs a quiet server: a concurrent long transaction (e.g. a 1M COPY) holds the xmin horizon back, VACUUM
    then cannot remove the deleted tuples yet and no tombstone is written (log_deletes=0, seen 2026-10-26)."""
    cur = table["cur"]
    x, ids = table["x"], table.get("ids", np.arange(table["x"].shape[0]))
    q = table["q"]
    victims = ids[:40].tolist()
    cur.execute("DELETE FROM tp_test WHERE id = ANY(%s)", (victims,))
    cur.execute("VACUUM (INDEX_CLEANUP ON) tp_test")  # PG skips index vacuuming when < 2 % of heap pages have dead items
    cur.execute("SELECT trigpoint_info('tp_test_ix')"); info = cur.fetchone()[0]
    assert "log_deletes=40" in info and "compactions=0" in info, info
    alive = ~np.isin(ids, victims)
    gt = _brute(x[alive], q)
    for qv, g in zip(q, gt):
        assert sorted(_search(cur, qv, {"trigpoint.exact": "on"})) == sorted(ids[alive][g].tolist())
    # force a compaction: threshold below the churn seen so far (30 inserts + 40 deletes on 12k rows)
    cur.execute("SET trigpoint.compact_frac = 0.001")
    cur.execute("VACUUM (INDEX_CLEANUP ON) tp_test")  # PG skips index vacuuming when < 2 % of heap pages have dead items
    cur.execute("RESET trigpoint.compact_frac")
    cur.execute("SELECT trigpoint_info('tp_test_ix')"); info = cur.fetchone()[0]
    assert "generation=1" in info and "compactions=1" in info and "log_inserts=0" in info, info
    for qv, g in zip(q, gt):
        assert sorted(_search(cur, qv, {"trigpoint.exact": "on"})) == sorted(ids[alive][g].tolist())
    with psycopg.connect(autocommit=True) as other:  # a backend that still holds generation 0? a fresh one loads 1
        oc = other.cursor()
        for qv, g in zip(q[:10], gt[:10]):
            assert sorted(_search(oc, qv, {"trigpoint.exact": "on"})) == sorted(ids[alive][g].tolist())
    cur.execute("RESET trigpoint.exact")
    table["x"], table["ids"] = x[alive], ids[alive]


@pytest.mark.shared_images
def test_shared_image_matches_owned_copy_and_is_listed(conn, table):
    """E1: the same session answers identically from the shared segment and from its own copy
    (trigpoint.shared_image on / off swap the cache entry), trigpoint_shared() lists the index's current
    generation once, and trigpoint_backend_memory() reports the kernel's RSS lines."""
    cur = table["cur"]
    q = table["q"][:20]
    cur.execute("RESET trigpoint.exact")
    res = {}
    for mode in ("off", "on", "off"):
        res.setdefault(mode, []).append([_search(cur, qv, {"trigpoint.shared_image": mode}) for qv in q])
    assert res["on"][0] == res["off"][0] == res["off"][1]
    cur.execute("SELECT trigpoint_info('tp_test_ix')"); gen = int(re.search(r"generation=(\d+)", cur.fetchone()[0]).group(1))
    cur.execute("SELECT generation, bytes, loading FROM trigpoint_shared() WHERE indexrelid = 'tp_test_ix'::regclass")
    listed = cur.fetchall()
    assert listed == [(gen, listed[0][1], False)] and listed[0][1] > 1_000_000, listed
    cur.execute("SELECT trigpoint_backend_memory()"); m = cur.fetchone()[0]
    assert "VmRSS=" in m and "RssAnon=" in m and "RssShmem=" in m, m
    cur.execute("RESET trigpoint.shared_image")


@pytest.mark.shared_images
def test_scan_in_flight_keeps_its_segment_across_a_compaction(conn, table):
    """E1 pin semantics: a cursor opened before a compaction keeps answering from the retired
    generation's segment (the DSM mapping is this backend's pin); the backend's next scan moves to
    the new generation and the old segment disappears from trigpoint_shared()."""
    cur = table["cur"]
    x, ids = table["x"], table.get("ids", np.arange(table["x"].shape[0]))
    q = table["q"]
    victims = ids[:5].tolist()
    cur.execute("DELETE FROM tp_test WHERE id = ANY(%s)", (victims,))
    alive = ~np.isin(ids, victims)
    gt = _brute(x[alive], q)
    cur.execute("SET trigpoint.shared_image = on"); cur.execute("SET trigpoint.exact = on")
    full = sorted(_search(cur, q[0]))
    assert full == sorted(ids[alive][gt[0]].tolist())
    cur.execute("SELECT generation FROM trigpoint_shared() WHERE indexrelid = 'tp_test_ix'::regclass"); gen0 = cur.fetchone()[0]
    cur.execute("BEGIN")
    cur.execute("DECLARE e1 CURSOR FOR SELECT id FROM tp_test ORDER BY v <-> %s::vector LIMIT 10", (_lit(q[0]),))
    cur.execute("FETCH 1 FROM e1"); first = [r[0] for r in cur.fetchall()]
    with psycopg.connect(autocommit=True) as other:  # compaction in another backend: the old generation is retired
        oc = other.cursor()
        oc.execute("SET trigpoint.compact_frac = 0.0001"); oc.execute("VACUUM (INDEX_CLEANUP ON) tp_test")
        oc.execute("SELECT trigpoint_info('tp_test_ix')"); info = oc.fetchone()[0]
        assert f"generation={gen0 + 1}" in info, info
        oc.execute("SELECT generation FROM trigpoint_shared() WHERE indexrelid = 'tp_test_ix'::regclass ORDER BY 1"); gens = [r[0] for r in oc.fetchall()]
        assert gens == [], gens  # the retired entry is gone (its segment lives while this cursor maps it); the next generation is loaded by its first scan
    cur.execute("FETCH ALL FROM e1"); rest = [r[0] for r in cur.fetchall()]
    assert sorted(first + rest) == full
    cur.execute("COMMIT")
    assert sorted(_search(cur, q[1])) == sorted(ids[alive][gt[1]].tolist())  # next scan: synced to the new generation
    cur.execute("SELECT generation FROM trigpoint_shared() WHERE indexrelid = 'tp_test_ix'::regclass ORDER BY 1"); gens = [r[0] for r in cur.fetchall()]
    assert gens == [gen0 + 1], gens
    cur.execute("RESET trigpoint.exact"); cur.execute("RESET trigpoint.shared_image")
    table["x"], table["ids"] = x[alive], ids[alive]


def test_inserts_concurrent_with_a_compaction_are_not_lost(conn, table):
    """Lost-write regression (found in E1, fixed after): rows inserted by another backend while a
    compaction runs — between its log sync and its meta flip — must be in the index afterwards. A thread
    inserts held-back corpus rows one per statement (autocommit) while this backend forces a compaction;
    afterwards every inserted id is found by an exact search for its own vector, in both backends."""
    import threading
    cur = table["cur"]
    x, ids = table["x"], table.get("ids", np.arange(table["x"].shape[0]))
    q = table["q"]
    base = int(ids.max()) + 1
    # rows not yet in the table: query vectors scaled a little (distinct from every corpus row)
    new = (q[:200] * 1.01).astype(np.float32)
    inserted = []
    stop = threading.Event()

    def inserter():
        with psycopg.connect(autocommit=True) as other:
            oc = other.cursor()
            i = 0
            while not stop.is_set() and i < len(new):
                oc.execute("INSERT INTO tp_test VALUES (%s, %s::vector)", (base + i, _lit(new[i])), prepare=False)
                inserted.append(base + i); i += 1

    cur.execute("DELETE FROM tp_test WHERE id = ANY(%s)", (ids[:3].tolist(),))  # churn above the forced threshold
    th = threading.Thread(target=inserter); th.start()
    time.sleep(0.05)
    cur.execute("SET trigpoint.compact_frac = 0.0001"); cur.execute("VACUUM (INDEX_CLEANUP ON) tp_test"); cur.execute("RESET trigpoint.compact_frac")
    stop.set(); th.join()
    cur.execute("SELECT trigpoint_info('tp_test_ix')"); info = cur.fetchone()[0]
    assert len(inserted) >= 5, (len(inserted), info)  # the compaction (~0.3 s at 12k rows) must have overlapped some inserts
    missing = []
    with psycopg.connect(autocommit=True) as other:
        oc = other.cursor()
        for c_ in (cur, oc):
            c_.execute("SET trigpoint.exact = on")
            for rid in inserted:
                c_.execute("SELECT id FROM tp_test ORDER BY v <-> %s::vector LIMIT 1", (_lit(new[rid - base]),), prepare=False)
                if c_.fetchone()[0] != rid: missing.append(rid)
            c_.execute("RESET trigpoint.exact")
    assert not missing, f"{len(missing)} of {len(inserted)} inserted rows not found after the compaction: {missing[:10]}; {info}"
    cur.execute("DELETE FROM tp_test WHERE id >= %s", (base,))
    alive = ~np.isin(ids, ids[:3])
    table["x"], table["ids"] = x[alive], ids[alive]


def test_inserts_during_create_index_concurrently_are_indexed(conn, table):
    """E2: rows inserted by another backend while CREATE INDEX CONCURRENTLY runs (the worker build) must be
    in the index afterwards — PostgreSQL's validate phase hands them to aminsert, which appends them to the
    log like any insert. Every inserted id is the top-1 exact result for its own vector."""
    import threading
    cur = table["cur"]
    q = table["q"]
    cur.execute("DROP TABLE IF EXISTS tp_cic")
    cur.execute("CREATE TABLE tp_cic WITH (autovacuum_enabled = off) AS SELECT id, v FROM tp_test")
    cur.execute("ALTER TABLE tp_cic ADD PRIMARY KEY (id)")
    cur.execute("SELECT max(id) FROM tp_cic"); base = cur.fetchone()[0] + 1
    new = (q[:300] * 1.02).astype(np.float32)
    inserted, stop = [], threading.Event()

    def inserter():
        with psycopg.connect(autocommit=True) as other:
            oc = other.cursor(); i = 0
            while not stop.is_set() and i < len(new):
                oc.execute("INSERT INTO tp_cic VALUES (%s, %s::vector)", (base + i, _lit(new[i])), prepare=False)
                inserted.append(base + i); i += 1

    th = threading.Thread(target=inserter); th.start()
    time.sleep(0.2)
    cur.execute("CREATE INDEX CONCURRENTLY tp_cic_ix ON tp_cic USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 200)")
    stop.set(); th.join()
    cur.execute("SELECT indisvalid FROM pg_index WHERE indexrelid = 'tp_cic_ix'::regclass"); assert cur.fetchone()[0]
    assert len(inserted) >= 5, len(inserted)
    cur.execute("SET trigpoint.exact = on"); cur.execute("SET enable_seqscan = off")
    missing = []
    for rid in inserted:
        cur.execute("SELECT id FROM tp_cic ORDER BY v <-> %s::vector LIMIT 1", (_lit(new[rid - base]),), prepare=False)
        if cur.fetchone()[0] != rid: missing.append(rid)
    cur.execute("RESET trigpoint.exact"); cur.execute("RESET enable_seqscan")
    assert not missing, f"{len(missing)} of {len(inserted)} rows inserted during CIC not found: {missing[:10]}"
    cur.execute("DROP TABLE tp_cic")


def test_repeated_compactions_keep_the_relation_bounded(conn, table):
    """E3: a retired generation's image and log pages return to the FSM after the flip and the next image is
    written into them; E3b (2026-11-02): the image is then moved into the freed pages and the relation truncated
    after its last live page, so five forced compactions leave the relation at one image plus the log (before
    E3b it sat at two images for good). WAL bytes per compaction are recorded."""
    cur = table["cur"]
    x, ids = table["x"], table.get("ids", np.arange(table["x"].shape[0]))
    q = table["q"]
    cur.execute("SET trigpoint.compact_frac = 0.0001")
    sizes, wal = [], []
    cur.execute("SELECT trigpoint_info('tp_test_ix')"); image_pages = int(re.search(r"image_pages=(\d+)", cur.fetchone()[0]).group(1))
    victims_all = []
    for k in range(5):
        victims = ids[k * 5:(k + 1) * 5].tolist(); victims_all += victims
        cur.execute("DELETE FROM tp_test WHERE id = ANY(%s)", (victims,))
        cur.execute("SELECT pg_current_wal_lsn()"); lsn0 = cur.fetchone()[0]
        cur.execute("VACUUM (INDEX_CLEANUP ON) tp_test")
        cur.execute("SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), %s)", (lsn0,)); wal.append(int(cur.fetchone()[0]))
        cur.execute("SELECT pg_relation_size('tp_test_ix') / 8192"); sizes.append(int(cur.fetchone()[0]))
    cur.execute("RESET trigpoint.compact_frac")
    cur.execute("SELECT trigpoint_info('tp_test_ix')"); info = cur.fetchone()[0]
    assert int(re.search(r"compactions=(\d+)", info).group(1)) >= 5, info
    # E3b: truncated after the last live page: never more than one image + the log + slack, and flat
    assert max(sizes) <= image_pages + 64, (sizes, image_pages)
    assert max(sizes[1:]) <= sizes[1] * 1.03 + 8, (sizes, image_pages)
    # each compaction logged about two images (the chain, then its relocation) plus the meta and the carried log
    assert all(w < (2 * image_pages + 64) * 8192 * 1.5 for w in wal), (wal, image_pages)
    alive = ~np.isin(ids, victims_all)
    gt = _brute(x[alive], q[:20])
    cur.execute("SET trigpoint.exact = on")
    for qv, g in zip(q[:20], gt):
        assert sorted(_search(cur, qv)) == sorted(ids[alive][g].tolist())
    cur.execute("RESET trigpoint.exact")
    table["x"], table["ids"] = x[alive], ids[alive]
    table["compaction_sizes"], table["compaction_wal"] = sizes, wal


def test_deleted_rows_after_vacuum_are_skipped(table):
    """The AM returns TIDs it found at build time; after DELETE + VACUUM those heap slots are dead
    or unused. The scan must neither crash nor return a deleted id, and exact mode must still find
    the true neighbours among the survivors."""
    cur = table["cur"]
    x, q = table["x"], table["q"]
    ids = table.get("ids", np.arange(x.shape[0]))
    gt = _brute(x, q)
    victims = sorted(set(ids[gt[:, 0]].tolist()))[:20]  # delete the nearest neighbour of the first queries
    cur.execute("DELETE FROM tp_test WHERE id = ANY(%s)", (victims,))
    cur.execute("VACUUM (INDEX_CLEANUP ON) tp_test")  # PG skips index vacuuming when < 2 % of heap pages have dead items  # S4: tombstones go to the index's log
    alive = ~np.isin(ids, victims)
    gt2 = _brute(x[alive], q); ids_alive = ids[alive]
    for qv, g in zip(q, gt2):
        got = _search(cur, qv, {"trigpoint.exact": "on"})
        assert not (set(got) & set(victims)), "a deleted row came back"
        assert sorted(got) == sorted(ids_alive[g].tolist())
    cur.execute("RESET trigpoint.exact")
    # and the calibrated path still answers without error
    assert len(_search(cur, q[0], {"trigpoint.probes": 8})) == 10
    table["x"], table["ids"] = x[alive], ids_alive


def test_levels_and_auto_cells_options(conn, table):
    cur = conn.cursor()
    D = table["x"].shape[1]
    lv = "8,16,32" if D < 128 else "16,64,128"
    cur.execute("DROP INDEX IF EXISTS tp_test_ix2")
    cur.execute(f"CREATE INDEX tp_test_ix2 ON tp_test USING trigpoint (v vector_l2_ops) WITH (levels = '{lv}', calibration_queries = 0)")
    cur.execute("SELECT trigpoint_info('tp_test_ix2')")
    info = cur.fetchone()[0]
    assert f"levels={lv}" in info and "ch_default=1.000,1.000,1.000" in info, info
    cur.execute("DROP INDEX tp_test_ix2")


def _own_table(conn, name, cells=64):
    """A fresh table + Trigpoint index from the module corpus, independent of the state the shared `tp_test` is in."""
    x, q = _corpus()
    n, D = x.shape
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {name}")
    cur.execute(f"CREATE TABLE {name} (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(n, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(n); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy(f"COPY {name} (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute(f"CREATE INDEX {name}_ix ON {name} USING trigpoint (v vector_l2_ops) WITH (cells = {cells}, calibration_queries = 500)")
    return cur, x, q


def _probe(conn, cur, name, xs, idss, q, settings):
    """Mean per-query cells_hit (trigpoint.debug NOTICE) and recall@10 against brute force over (xs, idss)."""
    notices = []
    handler = lambda d: notices.append(d.message_primary)  # noqa: E731
    conn.add_notice_handler(handler)
    try:
        for key, val in {**settings, "trigpoint.debug": "on", "enable_seqscan": "off"}.items():
            cur.execute(f"SET {key} = {val}")
        gt = _brute(xs, q)
        cells, rec = [], []
        for qv, g in zip(q, gt):
            notices.clear()
            cur.execute(f"SELECT id FROM {name} ORDER BY v <-> %s::vector LIMIT 10", (_lit(qv),), prepare=False)
            got = [r[0] for r in cur.fetchall()]
            m = [t for t in (re.search(r"cells_hit=(\d+)", n_) for n_ in notices) if t]
            assert m, notices
            cells.append(int(m[-1].group(1)))
            rec.append(len(set(got) & set(idss[g].tolist())) / 10)
    finally:
        conn.remove_notice_handler(handler)
        cur.execute("RESET trigpoint.debug")
    return float(np.mean(cells)), float(np.mean(rec))


def _live_inserts_check(conn, k_inserts, flat_max=None):
    cur, x, q = _own_table(conn, "tp_live")
    settings = {"trigpoint.hits": 200, "trigpoint.probes": 16}
    if flat_max is not None:
        settings["trigpoint.delta_flat_max"] = flat_max
    try:
        ids = np.arange(x.shape[0])
        cells_clean, rec_clean = _probe(conn, cur, "tp_live", x, ids, q, settings)
        rng = np.random.default_rng(11)
        pick = rng.integers(0, x.shape[0], k_inserts)
        new = (x[pick] + 0.02 * rng.normal(size=(k_inserts, x.shape[1]))).astype(np.float32)
        base = 3 * 10 ** 6
        for i, v in enumerate(new):
            cur.execute("INSERT INTO tp_live VALUES (%s, %s::vector)", (base + i, _lit(v)), prepare=False)
        cells_live, rec_live = _probe(conn, cur, "tp_live", np.concatenate([x, new]), np.concatenate([ids, base + np.arange(k_inserts)]), q, settings)
        print(f"{k_inserts} live inserts: cells_hit clean {cells_clean:.1f} -> live {cells_live:.1f}; recall {rec_clean:.3f} -> {rec_live:.3f}")
        assert cells_live <= cells_clean + 2.0, f"live inserts widened the router: cells_hit {cells_clean:.1f} -> {cells_live:.1f}"
        assert rec_live >= rec_clean - 0.02, f"recall {rec_clean:.3f} -> {rec_live:.3f} with {k_inserts} live inserts"
    finally:
        for key in settings:
            cur.execute(f"RESET {key}")
        cur.execute("DROP TABLE IF EXISTS tp_live")


def test_live_inserts_do_not_widen_the_router(conn, table):
    """Regression for the pending-log recall loss (2026-10-28, core fix 2f331ff): the router's hits are the top M
    of the base graph's and the delta graph's results together. Before the fix the delta's own M hits were appended,
    so a few hundred live inserts marked up to M extra (random, far) cells, which displaced the centroid fill of
    the probe list and cost 4 pt of recall at 1M with 2,000 live entries. Here 300 live near-copies of random rows
    (fewer than the delta graph's first capacity of 1,024 nodes) must leave the per-query cells_hit (trigpoint.debug
    NOTICE) and the recall within a margin of the clean values on the same queries. Own table: the shared tp_test
    has been compacted by the tests above, so its values are not comparable across the module."""
    _live_inserts_check(conn, 300)


def test_live_inserts_beyond_the_delta_graphs_first_capacity_do_not_widen_the_router(conn, table):
    """1,500 live inserts cross the delta graph's first capacity (1,024 nodes), so the graph is grown through
    tp_graph_private_copy. Before the fix of 2026-10-28 the copy squared the weights again (step^4), the grown
    delta's scores were mis-scaled against the base graph's and the cell list widened 1.5x."""
    _live_inserts_check(conn, 1500)


def test_live_inserts_through_a_linked_delta_graph(conn, table):
    """H17 (2026-10-29): below trigpoint.delta_flat_max (default 50,000) the appended positions are scored flat by
    the router and never linked. With the threshold at 0 every one is linked as it arrives (the behaviour before
    H17); the same 1,500 inserts must route as well through the graph."""
    _live_inserts_check(conn, 1500, flat_max=0)


def test_live_inserts_crossing_the_flat_threshold_link_the_delta_once(conn, table):
    """H17: with the threshold at 1,000 the 1,001st appended position links the 1,000 flat ones at once and every
    later one as it arrives; the router must read the same cells and recall as with a flat or a linked delta."""
    _live_inserts_check(conn, 1500, flat_max=1000)


def test_cancelled_log_pairs_are_not_replayed_into_the_delta_graph(conn, table):
    """A backend's first scan replays the whole log since the image into its private delta graph, about 1 ms per
    INSERT record (10-22 s for 43,238 records on the 1M table, finding 3 of 2026-10-28). INSERT records whose DELETE
    is in the same replay now get their position and tombstone but no graph node. Here 1,500 rows are inserted and
    deleted below the compaction threshold; a fresh backend must report them as cancelled in its load NOTICE, read
    the clean recall and cells_hit, and still route 200 further live inserts."""
    cur, x, q = _own_table(conn, "tp_replay")
    settings = {"trigpoint.hits": 200, "trigpoint.probes": 16}
    try:
        ids = np.arange(x.shape[0])
        cells_clean, rec_clean = _probe(conn, cur, "tp_replay", x, ids, q, settings)
        rng = np.random.default_rng(23)
        k = 1500
        new = (x[rng.integers(0, x.shape[0], k)] + 0.02 * rng.normal(size=(k, x.shape[1]))).astype(np.float32)
        base = 4 * 10 ** 6
        for i, v in enumerate(new):
            cur.execute("INSERT INTO tp_replay VALUES (%s, %s::vector)", (base + i, _lit(v)), prepare=False)
        cur.execute("DELETE FROM tp_replay WHERE id >= %s", (base,))
        cur.execute("SET trigpoint.compact_frac = 100"); cur.execute("VACUUM (INDEX_CLEANUP ON) tp_replay"); cur.execute("RESET trigpoint.compact_frac")
        cur.execute("SELECT trigpoint_info('tp_replay_ix')"); info = cur.fetchone()[0]
        assert f"log_inserts={k} log_deletes={k} compactions=0" in info, info
        with psycopg.connect(autocommit=True) as other:
            notices = []
            other.add_notice_handler(lambda d: notices.append(d.message_primary))
            ocur = other.cursor()
            for key, val in {**settings, "trigpoint.debug": "on", "enable_seqscan": "off"}.items():
                ocur.execute(f"SET {key} = {val}")
            ocur.execute("SELECT id FROM tp_replay ORDER BY v <-> %s::vector LIMIT 10", (_lit(q[0]),), prepare=False)
            ocur.fetchall()
            m = [t for t in (re.search(r"replay (\d+) records .*?\((\d+) cancelled", n_) for n_ in notices) if t]
            assert m, notices
            assert (int(m[0].group(1)), int(m[0].group(2))) == (2 * k, k), m[0].group(0)
            ocur.execute("RESET trigpoint.debug")
            cells_fresh, rec_fresh = _probe(other, ocur, "tp_replay", x, ids, q, settings)
            print(f"cancelled pairs: cells_hit clean {cells_clean:.1f} -> fresh backend {cells_fresh:.1f}; recall {rec_clean:.3f} -> {rec_fresh:.3f}")
            assert abs(cells_fresh - cells_clean) <= 2.0 and rec_fresh >= rec_clean - 0.02
            k2 = 200
            new2 = (x[rng.integers(0, x.shape[0], k2)] + 0.02 * rng.normal(size=(k2, x.shape[1]))).astype(np.float32)
            for i, v in enumerate(new2):
                ocur.execute("INSERT INTO tp_replay VALUES (%s, %s::vector)", (base + k + i, _lit(v)), prepare=False)
            cells_live, rec_live = _probe(other, ocur, "tp_replay", np.concatenate([x, new2]), np.concatenate([ids, base + k + np.arange(k2)]), q, settings)
            assert cells_live <= cells_clean + 2.0 and rec_live >= rec_clean - 0.02, (cells_live, rec_live)
            for key in settings:
                ocur.execute(f"RESET {key}")
    finally:
        for key in settings:
            cur.execute(f"RESET {key}")
        cur.execute("DROP TABLE IF EXISTS tp_replay")


def test_calibrated_path_survives_a_compaction(conn, table):
    """The compaction builds the next image's router graph through tp_graph_private_copy. With the squared
    weights (before 2026-10-28) the beam ranked by node norm, query-independent: cells_hit a constant and recall
    1.0 -> 0.21 on this table, exact mode unaffected."""
    cur, x, q = _own_table(conn, "tp_cmp")
    settings = {"trigpoint.hits": 200, "trigpoint.probes": 16}
    try:
        ids = np.arange(x.shape[0])
        cells0, rec0 = _probe(conn, cur, "tp_cmp", x, ids, q, settings)
        victims = ids[:5].tolist()
        cur.execute("DELETE FROM tp_cmp WHERE id = ANY(%s)", (victims,))
        cur.execute("SET trigpoint.compact_frac = 0.0001"); cur.execute("VACUUM (INDEX_CLEANUP ON) tp_cmp"); cur.execute("RESET trigpoint.compact_frac")
        cur.execute("SELECT trigpoint_info('tp_cmp_ix')"); info = cur.fetchone()[0]
        assert "compactions=1" in info, info
        alive = ~np.isin(ids, victims)
        cells1, rec1 = _probe(conn, cur, "tp_cmp", x[alive], ids[alive], q, settings)
        print(f"compaction: cells_hit {cells0:.1f} -> {cells1:.1f}; recall {rec0:.3f} -> {rec1:.3f}")
        assert rec1 >= rec0 - 0.02, f"the calibrated path lost recall across a compaction: {rec0:.3f} -> {rec1:.3f}"
    finally:
        for key in settings:
            cur.execute(f"RESET {key}")
        cur.execute("DROP TABLE IF EXISTS tp_cmp")
