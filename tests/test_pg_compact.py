"""Hardening H13: a compaction keeps only the live rows. Under churn (inserts + deletes + VACUUM) the image
must stay the size of the live rows, the router must keep its recall, and the renumbered ids must stay
consistent (every live row found at exact = on, deleted rows never returned, trigpoint_check deep clean,
records appended during the compaction carried with their new ids).
"""

import struct

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _brute, _lit  # noqa: E402

D = 32


def _connect():
    try:
        c = psycopg.connect(autocommit=True, connect_timeout=3)
        c.execute("SET statement_timeout = '%ds'" % (300 * (5 if 'sanitizer build' in c.execute('SELECT version()').fetchone()[0] else 1)))
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


def corpus(n, seed):
    rng = np.random.default_rng(seed)
    cents = rng.normal(size=(32, D)).astype(np.float32) * 3
    return (cents[rng.integers(0, 32, n)] + rng.normal(size=(n, D))).astype(np.float32)


def info(cur, index):
    cur.execute(f"SELECT trigpoint_info('{index}')")
    s = cur.fetchone()[0]
    out = {}
    for t in s.split():
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    cur.execute(f"SELECT pg_relation_size('{index}')")
    out["bytes"] = cur.fetchone()[0]
    return out


def check(cur, index):
    cur.execute(f"SELECT * FROM trigpoint_check('{index}', true)")
    lines = [r[0] for r in cur.fetchall()]
    return int(lines[-1].split(" problems")[0].rsplit(" ", 1)[1]), lines


def knn(cur, table, q, k=10, exact=False):
    cur.execute(f"SET trigpoint.exact = {'on' if exact else 'off'}")
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SELECT id FROM {table} ORDER BY v <-> %s::vector LIMIT {k}", (_lit(q),), prepare=False)
    out = [r[0] for r in cur.fetchall()]
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET trigpoint.exact = off")
    return out


def recall(cur, table, live, qs):
    ids = np.array(sorted(live))
    x = np.stack([live[i] for i in ids])
    hits = 0
    for q in qs:
        true = set(ids[_brute(x, q[None, :])[0]].tolist())
        hits += len(set(knn(cur, table, q)) & true)
    return hits / (10 * len(qs))


def test_churn_keeps_the_image_at_the_live_rows(conn):
    cur = conn.cursor()
    x = corpus(6000, 0)
    cur.execute("DROP TABLE IF EXISTS tp_cmp")
    cur.execute(f"CREATE TABLE tp_cmp (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(len(x), dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(len(x)); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_cmp (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("CREATE INDEX tp_cmp_ix ON tp_cmp USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    live = {i: x[i] for i in range(len(x))}
    fresh = info(cur, "tp_cmp_ix")
    rng = np.random.default_rng(1)
    qs = corpus(100, 9) + 0.3 * rng.normal(size=(100, D)).astype(np.float32)
    r0 = recall(cur, "tp_cmp", live, qs)
    next_id = len(x)
    for rnd in range(5):
        new = corpus(2000, 10 + rnd)
        for v in new:
            cur.execute("INSERT INTO tp_cmp VALUES (%s, %s::vector)", (next_id, _lit(v)), prepare=False)
            live[next_id] = v
            next_id += 1
        gone = rng.choice(sorted(live), 2000, replace=False).tolist()
        cur.execute("DELETE FROM tp_cmp WHERE id = ANY(%s)", (gone,))
        for i in gone:
            live.pop(i)
        cur.execute("VACUUM tp_cmp")
        meta = info(cur, "tp_cmp_ix")
        assert int(meta["compactions"]) == rnd + 1, meta
        assert int(meta["n"]) == len(live) == 6000, (rnd, meta["n"], len(live))
        assert int(meta["n_image"]) == len(live) and meta["log_inserts"] == "0" and meta["log_deletes"] == "0", meta
        # the image is the live rows' size, and (E3b) the relation is one image plus the log: the retired
        # generation's pages are reused by the relocated image and the tail is truncated
        assert int(meta["image_pages"]) <= 1.3 * int(fresh["image_pages"]), (rnd, meta["image_pages"], fresh["image_pages"])
        assert meta["bytes"] <= 1.3 * fresh["bytes"], (rnd, meta["bytes"], fresh["bytes"])
        problems, lines = check(cur, "tp_cmp_ix")
        assert problems == 0, "\n".join(lines)
        # every live row is found by its own vector at exact = on, no deleted row ever comes back
        sample = rng.choice(sorted(live), 100, replace=False).tolist()
        for i in sample:
            assert knn(cur, "tp_cmp", live[i], k=1, exact=True) == [i], (rnd, i)
        for i in gone[:50]:
            got = knn(cur, "tp_cmp", x[i] if i < len(x) else live.get(i, qs[0]), k=5)
            assert i not in got, (rnd, i)
        r = recall(cur, "tp_cmp", live, qs)
        assert r >= r0 - 0.03, (rnd, r, r0)
    # the compacted index is within a few points of a fresh build over the same rows
    r_before = recall(cur, "tp_cmp", live, qs)
    cur.execute("REINDEX INDEX tp_cmp_ix")
    r_fresh = recall(cur, "tp_cmp", live, qs)
    assert r_before >= r_fresh - 0.03, (r_before, r_fresh)
    cur.execute("DROP TABLE tp_cmp")


def test_compaction_recalibrates_after_drift(conn):
    """H16: rows that drift from the build's distribution (clustered around existing rows instead of the
    corpus centres) made the calibrated bound over-prune; a compaction re-fits the tables over the live rows."""
    cur = conn.cursor()
    x = corpus(6000, 20)
    rng = np.random.default_rng(21)
    drift_cents = x[rng.choice(6000, 32, replace=False)]
    xd = (drift_cents[rng.integers(0, 32, 6000)] + rng.normal(size=(6000, D))).astype(np.float32)

    def build(table, recal):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
        dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
        rec = np.empty(len(x), dt)
        rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(len(x)); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
        with cur.copy(f"COPY {table} (id, v) FROM STDIN (FORMAT BINARY)") as cp:
            cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
        cur.execute(f"CREATE INDEX {table}_ix ON {table} USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
        # replace the corpus by the drifted rows
        for i, v in enumerate(xd):
            cur.execute(f"INSERT INTO {table} VALUES (%s, %s::vector)", (10000 + i, _lit(v)), prepare=False)
        cur.execute(f"DELETE FROM {table} WHERE id < 6000")
        cur.execute(f"SET trigpoint.recalibrate_queries = {recal}")
        cur.execute(f"VACUUM {table}")
        cur.execute("RESET trigpoint.recalibrate_queries")
        problems, lines = check(cur, f"{table}_ix")
        assert problems == 0, "\n".join(lines)

    qs = xd[:150] + 0.3 * rng.normal(size=(150, D)).astype(np.float32)
    live = {10000 + i: xd[i] for i in range(6000)}
    build("tp_drift_off", 0)
    r_off = recall(cur, "tp_drift_off", live, qs)
    build("tp_drift_on", 500)
    r_on = recall(cur, "tp_drift_on", live, qs)
    cur.execute("REINDEX INDEX tp_drift_on_ix")
    r_fresh = recall(cur, "tp_drift_on", live, qs)
    print(f"\ndrift recall: build tables kept {r_off:.3f}, re-calibrated at compaction {r_on:.3f}, fresh REINDEX {r_fresh:.3f}")
    assert r_on >= r_fresh - 0.02, (r_on, r_fresh)      # the compacted index calibrates like a fresh build
    assert r_on >= r_off - 0.005, (r_on, r_off)         # and never worse than the build's tables (the soak with --drift shows the gap: 0.91 vs 0.99)
    cur.execute("DROP TABLE tp_drift_off, tp_drift_on")


def test_recalibration_over_the_first_cells_matches_every_cell(conn):
    """H18 (2026-10-29): the compaction's calibration queries open only the first trigpoint.recalibrate_probes
    cells in the router's order (default 64) instead of every cell; at 1M x 1536 the exact form took 171 s of a
    321 s VACUUM. The tables must come out as with every cell: c_h within 0.01 at each level."""
    cur = conn.cursor()
    x = corpus(8000, 30)
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(len(x), dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(len(x)); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    ch = {}
    for probes in (0, 8):
        cur.execute("DROP TABLE IF EXISTS tp_recal")
        cur.execute(f"CREATE TABLE tp_recal (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
        with cur.copy("COPY tp_recal (id, v) FROM STDIN (FORMAT BINARY)") as cp:
            cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
        cur.execute("CREATE INDEX tp_recal_ix ON tp_recal USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
        cur.execute("DELETE FROM tp_recal WHERE id % 4 = 0")
        cur.execute(f"SET trigpoint.recalibrate_probes = {probes}")
        cur.execute("VACUUM tp_recal")
        cur.execute("RESET trigpoint.recalibrate_probes")
        meta = info(cur, "tp_recal_ix")
        assert meta["compactions"] == "1", meta
        ch[probes] = [float(t) for t in meta["ch_default"].split(",")]
        problems, lines = check(cur, "tp_recal_ix")
        assert problems == 0, "\n".join(lines)
    print(f"\nc_h every cell {ch[0]}, first 8 of 64 cells {ch[8]}")
    assert all(abs(a - b) <= 0.01 for a, b in zip(ch[0], ch[8])), ch
    cur.execute("DROP TABLE tp_recal")


def test_compaction_returns_the_space(conn):
    """E3b (2026-11-02): the compaction moves its new image into the pages the retired generation freed and
    truncates the relation after its last live page. Before: the first compaction doubled the file (1M x 1,536:
    856 MB at build, 1,633 MB after VACUUM with the image at 767 MB) and no later one shrank it. Now, after
    10 % and then 25 % of the rows are deleted and vacuumed, the relation is at most the
    image plus the log and never above the build; the deep check is clean, exact search finds every live row,
    and a second backend holding the index open only postpones the truncation to the next VACUUM."""
    cur = conn.cursor()
    x = corpus(20000, 21)
    cur.execute("DROP TABLE IF EXISTS tp_rec")
    cur.execute(f"CREATE TABLE tp_rec (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(len(x), dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(len(x)); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_rec (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("CREATE INDEX tp_rec_ix ON tp_rec USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 200)")
    build = info(cur, "tp_rec_ix")
    assert int(build["rel_pages"]) == int(build["image_pages"]) + 1, build
    live = {i: x[i] for i in range(len(x))}
    rng = np.random.default_rng(3)
    qs = corpus(50, 8)
    cur.execute("SET trigpoint.compact_frac = 0.05")  # 10 % of 20k rows is 2,000 records, below compact_records; at 1M the record count triggers
    for stage, upto in enumerate((2000, 5000), 1):
        cur.execute("DELETE FROM tp_rec WHERE id < %s", (upto,))
        for i in range(upto):
            live.pop(i, None)
        cur.execute("VACUUM tp_rec")
        meta = info(cur, "tp_rec_ix")
        assert int(meta["compactions"]) == stage and int(meta["n"]) == len(live), meta
        # the image shrank with the rows, the file is the image + the meta page + a few log pages, below the build
        assert int(meta["image_pages"]) < int(build["image_pages"]), (stage, meta["image_pages"], build["image_pages"])
        assert int(meta["rel_pages"]) <= int(meta["image_pages"]) + 1 + 8, (stage, meta["rel_pages"], meta["image_pages"])
        assert meta["bytes"] < build["bytes"], (stage, meta["bytes"], build["bytes"])
        problems, lines = check(cur, "tp_rec_ix")
        assert problems == 0, "\n".join(lines)
        for i in rng.choice(sorted(live), 50, replace=False).tolist():
            assert knn(cur, "tp_rec", live[i], k=1, exact=True) == [i], (stage, i)
        for q in qs[:10]:
            assert not (set(knn(cur, "tp_rec", q)) & set(range(upto))), stage
    # a backend holding the index open (a transaction that read it after the DELETE committed, so it holds the
    # index's AccessShareLock but no snapshot that keeps the dead rows) blocks the AccessExclusiveLock: the
    # compaction still happens, the truncation is skipped, and the next VACUUM with the index free truncates
    cur.execute("DELETE FROM tp_rec WHERE id < 8000")
    for i in range(8000):
        live.pop(i, None)
    other = _connect(); oc = other.cursor()
    oc.execute("BEGIN"); oc.execute("SET LOCAL enable_seqscan = off")
    oc.execute("SELECT id FROM tp_rec ORDER BY v <-> %s::vector LIMIT 1", (_lit(qs[0]),), prepare=False); oc.fetchall()
    cur.execute("SET trigpoint.compact_frac = 0.0001")
    cur.execute("VACUUM tp_rec")
    held = info(cur, "tp_rec_ix")
    assert int(held["compactions"]) == 3, held
    assert int(held["rel_pages"]) > int(held["image_pages"]) + 1 + 8, held  # the moved pages' old copies wait at the tail
    oc.execute("COMMIT"); other.close()
    cur.execute("DELETE FROM tp_rec WHERE id < 8010")
    for i in range(8010):
        live.pop(i, None)
    cur.execute("VACUUM (INDEX_CLEANUP ON) tp_rec")  # ten dead rows: below the executor's index-vacuum bypass without the option
    cur.execute("RESET trigpoint.compact_frac")
    freed = info(cur, "tp_rec_ix")
    assert int(freed["compactions"]) == 4, freed
    assert int(freed["rel_pages"]) <= int(freed["image_pages"]) + 1 + 8, (held["rel_pages"], freed["rel_pages"], freed["image_pages"])
    assert int(freed["rel_pages"]) <= int(held["rel_pages"]), (held["rel_pages"], freed["rel_pages"])
    problems, lines = check(cur, "tp_rec_ix")
    assert problems == 0, "\n".join(lines)
    for i in rng.choice(sorted(live), 50, replace=False).tolist():
        assert knn(cur, "tp_rec", live[i], k=1, exact=True) == [i], i
    cur.execute("DROP TABLE tp_rec")
