"""Hardening H3: trigpoint_check(regclass, deep) and the load path's refusal of corrupt pages.

Runs against a reachable PostgreSQL with the extension (PGHOST etc.; the local loop is
docker/compose.local.yml on port 5433), skipped otherwise. The corruption cases write single bytes into
index pages through trigpoint_debug_poke (superuser, trigpoint.debug = on), run the checker, open the index
on the owned path in a fresh backend (shared_image = off, so the pages are read again) and assert that the
server answers with an ERROR or a correct result, never a crash: the module's own connection must survive
every case. Targeted cases also assert that the checker names the problem.
"""

import os
import random
import struct
import time

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from test_pg_extension import _brute, _corpus, _lit  # noqa: E402

BLCKSZ = 8192
PAGE_HEADER = 24            # SizeOfPageHeaderData; the meta page's contents and the line pointers start here
SPECIAL = BLCKSZ - 16       # MAXALIGN(sizeof(TrigpointPageOpaqueData)): kind u16, unused u16, generation u32, next u32
CHUNK = 8144                # TP_CHUNK_BYTES
# TrigpointMetaPageData field offsets (C ABI, see pgext/trigpoint.h)
META = {"magic": 0, "version": 4, "n": 8, "D": 12, "C": 16, "L": 20, "nchunks": 48, "blob_bytes": 56, "core_bytes": 64,
        "generation": 72, "image_start": 76, "log_first": 80, "log_last": 84, "n_image": 104}
# FlatHeader field offsets (csrc/flat_image.h)
FLAT = {"magic": 0, "version": 4, "n": 8, "D": 12, "C": 16, "L": 20, "lv0": 24, "M": 40, "npos_of": 48, "total_bytes": 104,
        "lo": 176, "step": 184, "proj": 192, "mean": 200, "cent": 208, "cal0": 216, "perm": 248, "pos_of": 256, "cell_of": 264,
        "deleted": 272, "err0": 280, "tail0": 312, "blk_code_off": 344, "blk_pos_off": 352, "blk_codes": 360, "blk_pos": 368,
        "graph": 376, "graph_bytes": 384}


def _connect(**kw):
    try:
        return psycopg.connect(autocommit=True, connect_timeout=3, **kw)
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
    cur.execute("SET statement_timeout = '%ds'" % (120 * (5 if 'sanitizer build' in cur.execute('SELECT version()').fetchone()[0] else 1)))  # a hang is a failure, not a wait (the barrier bug of 2026-10-28)
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'trigpoint'")
    if cur.fetchone()[0] < "0.3":
        pytest.skip("trigpoint < 0.3 has no trigpoint_check")
    yield c
    c.close()


@pytest.fixture(scope="module")
def table(conn):
    x, q = _corpus()
    n, D = x.shape
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS tp_chk")
    cur.execute(f"CREATE TABLE tp_chk (id int PRIMARY KEY, v vector({D})) WITH (autovacuum_enabled = off)")
    dt = np.dtype([("ncols", ">i2"), ("l1", ">i4"), ("id", ">i4"), ("l2", ">i4"), ("dim", ">u2"), ("unused", ">u2"), ("v", ">f4", (D,))])
    rec = np.empty(n, dt)
    rec["ncols"] = 2; rec["l1"] = 4; rec["id"] = np.arange(n); rec["l2"] = 4 + 4 * D; rec["dim"] = D; rec["unused"] = 0; rec["v"] = x
    with cur.copy("COPY tp_chk (id, v) FROM STDIN (FORMAT BINARY)") as cp:
        cp.write(b"PGCOPY\n\xff\r\n\0" + struct.pack(">ii", 0, 0)); cp.write(rec.tobytes()); cp.write(struct.pack(">h", -1))
    cur.execute("CREATE INDEX tp_chk_ix ON tp_chk USING trigpoint (v vector_l2_ops) WITH (cells = 64, calibration_queries = 500)")
    yield {"x": x, "q": q, "gt": _brute(x, q), "cur": cur, "n": n, "D": D}
    cur.execute("DROP TABLE IF EXISTS tp_chk")


def check(cur, deep=False):
    cur.execute("SELECT * FROM trigpoint_check('tp_chk_ix', %s)", (deep,))
    lines = [r[0] for r in cur.fetchall()]
    assert lines and lines[-1].startswith("summary:"), lines
    problems = int(lines[-1].split(" problems")[0].rsplit(" ", 1)[1])
    return problems, lines


class Pager:
    """peek / poke through the debug hooks on its own superuser connection with trigpoint.debug on"""

    def __init__(self):
        self.c = _connect()
        self.cur = self.c.cursor()
        self.cur.execute("SET trigpoint.debug = on")
        self.cur.execute("SET client_min_messages = warning")

    def peek(self, blk, off):
        self.cur.execute("SELECT trigpoint_debug_peek('tp_chk_ix', %s, %s)", (blk, off))
        return self.cur.fetchone()[0]

    def poke(self, blk, off, val):
        self.cur.execute("SELECT trigpoint_debug_poke('tp_chk_ix', %s, %s, %s)", (blk, off, val))

    def u16(self, blk, off):
        return self.peek(blk, off) | (self.peek(blk, off + 1) << 8)

    def u32(self, blk, off):
        return self.u16(blk, off) | (self.u16(blk, off + 2) << 16)

    def meta(self, field):
        return self.u32(0, PAGE_HEADER + META[field])

    def nblocks(self):
        self.cur.execute("SELECT pg_relation_size('tp_chk_ix') / 8192")
        return int(self.cur.fetchone()[0])

    def item1(self, blk):
        """(offset, length) of line pointer 1 on a page"""
        lp = self.u32(blk, PAGE_HEADER)
        return lp & 0x7FFF, lp >> 17

    def flat_field(self, field):
        """(blk, off) of a FlatHeader field: the flat image follows n_image TIDs in the chunked blob"""
        boff = self.meta("n_image") * 8 + FLAT[field]
        blk = self.meta("image_start") + boff // CHUNK
        item_off, _ = self.item1(blk)
        return blk, item_off + boff % CHUNK

    def close(self):
        self.c.close()


def owned_query(x, gt, settings="SET trigpoint.shared_image = off"):
    """Query on a fresh backend that reads the pages itself. Returns ('ok', recall) or ('error', message);
    anything else (a dropped connection) is a crash and raises."""
    c = _connect()
    try:
        cur = c.cursor()
        cur.execute(settings)
        cur.execute("SET trigpoint.exact = off")
        hits = 0
        for i in range(5):
            cur.execute("SELECT id FROM tp_chk ORDER BY v <-> %s::vector LIMIT 10", (_lit(x[i]),), prepare=False)
            hits += len(set(r[0] for r in cur.fetchall()) & set(gt[i].tolist()))
        return "ok", hits / 50
    except psycopg.errors.OperationalError as e:
        if c.closed or "closed the connection" in str(e):
            raise AssertionError(f"backend crashed: {e}")
        return "error", str(e)
    except psycopg.Error as e:
        return "error", str(e)
    finally:
        c.close()


def alive(conn):
    cur = conn.cursor()
    cur.execute("SELECT 1")
    return cur.fetchone()[0] == 1


def test_check_is_clean_on_a_fresh_index(table):
    problems, lines = check(table["cur"], deep=True)
    assert problems == 0, "\n".join(lines)
    assert any(l.startswith("ok: flat image: deep") for l in lines), lines
    assert any(l.startswith("ok: TID table") for l in lines), lines


def test_check_is_clean_after_inserts_deletes_and_vacuum(table):
    cur, x = table["cur"], table["x"]
    rng = np.random.default_rng(1)
    for i in range(300):
        cur.execute("INSERT INTO tp_chk VALUES (%s, %s::vector)", (100000 + i, _lit(x[i] + 0.01 * rng.normal(size=x.shape[1]))))
    problems, lines = check(cur, deep=True)
    assert problems == 0, "\n".join(lines)
    assert any("300 inserts" in l for l in lines), lines
    cur.execute("DELETE FROM tp_chk WHERE id >= 100000 AND id < 100200")
    cur.execute("VACUUM tp_chk")
    problems, lines = check(cur, deep=True)
    assert problems == 0, "\n".join(lines)
    cur.execute("DELETE FROM tp_chk WHERE id >= 100000")
    cur.execute("VACUUM tp_chk")
    assert check(cur, deep=True)[0] == 0


def test_check_refuses_a_non_trigpoint_index(table):
    with pytest.raises(psycopg.Error, match="not a trigpoint index"):
        table["cur"].execute("SELECT * FROM trigpoint_check('tp_chk_pkey')")


def test_page_hooks_need_debug_and_superuser(table):
    cur = table["cur"]
    with pytest.raises(psycopg.Error, match="trigpoint.debug"):
        cur.execute("SELECT trigpoint_debug_peek('tp_chk_ix', 0, 0)")


def test_targeted_corruptions_are_reported_and_refused(conn, table):
    """Each case: poke, the checker reports it, a fresh owned-path backend either errors or answers correctly,
    nobody crashes, the byte goes back and the checker is clean again."""
    cur, x, gt = table["cur"], table["x"], table["gt"]
    pg = Pager()
    # a few live log records so the log cases have a page to hit
    rng = np.random.default_rng(2)
    for i in range(20):
        cur.execute("INSERT INTO tp_chk VALUES (%s, %s::vector)", (200000 + i, _lit(x[i] + 0.01 * rng.normal(size=x.shape[1]))))
    assert check(cur)[0] == 0
    log_blk = pg.meta("log_first")
    assert log_blk != 0xFFFFFFFF
    rec_off, _ = pg.item1(log_blk)
    img = pg.meta("image_start")
    cases = [
        ("meta magic", 0, PAGE_HEADER + META["magic"], 0x00, "magic"),
        ("meta version", 0, PAGE_HEADER + META["version"], 0x09, "format"),
        ("meta blob_bytes", 0, PAGE_HEADER + META["blob_bytes"] + 1, 0x7F, "inconsistent"),
        ("meta image_start beyond the relation", 0, PAGE_HEADER + META["image_start"] + 2, 0x7F, "inconsistent"),
        ("meta n_image", 0, PAGE_HEADER + META["n_image"], 0x01, ""),
        ("image page kind", img, SPECIAL, 0x05, "not an image page"),
        ("image page generation", img, SPECIAL + 4, 0x7E, "not an image page"),
        ("image page next pointer", img, SPECIAL + 8 + 3, 0x7F, "image chain"),
        ("image item length high byte", img, PAGE_HEADER + 3, 0x7F, "image page"),
        ("flat header magic", *pg.flat_field("magic"), 0x00, "magic"),
        ("flat header L", *pg.flat_field("L"), 0x09, "out of range"),
        ("flat header proj offset", *(lambda b, o: (b, o + 7))(*pg.flat_field("proj")), 0x7F, "section proj"),
        ("flat header perm offset", *(lambda b, o: (b, o + 6))(*pg.flat_field("perm")), 0x7F, "section perm"),
        ("flat header graph offset", *(lambda b, o: (b, o + 7))(*pg.flat_field("graph")), 0x7F, "section graph"),
        ("flat header blk_pos_off offset", *(lambda b, o: (b, o + 7))(*pg.flat_field("blk_pos_off")), 0x7F, "blk_pos_off"),
        ("flat header npos_of below n", *pg.flat_field("npos_of"), 0x00, "npos_of"),
        ("log record kind", log_blk, rec_off, 0x07, "kind"),
        ("log record cell", log_blk, rec_off + 2 + 1, 0xFF, "cell"),
        ("log record id", log_blk, rec_off + 4 + 3, 0x7F, "id"),
        ("log page kind", log_blk, SPECIAL, 0x09, "not a log page"),
    ]
    for name, blk, off, val, expect in cases:
        orig = pg.peek(blk, off)
        if orig == val:
            val = (val + 1) & 0xFF
        pg.poke(blk, off, val)
        try:
            problems, lines = check(cur)
            assert problems >= 1, f"{name}: checker saw nothing\n" + "\n".join(lines)
            if expect:
                assert any(expect in l for l in lines if l.startswith("problem:")), f"{name}: expected {expect!r} in\n" + "\n".join(lines)
            status, detail = owned_query(x, gt)
            assert status in ("ok", "error"), (name, status, detail)
            if status == "ok":
                assert detail >= 0.9, (name, detail)
            assert alive(conn), name
        finally:
            pg.poke(blk, off, orig)
        problems, lines = check(cur)
        assert problems == 0, f"{name}: not clean after restoring the byte\n" + "\n".join(lines)
    cur.execute("DELETE FROM tp_chk WHERE id >= 200000")
    cur.execute("VACUUM tp_chk")
    pg.close()


def test_random_corruptions_never_crash(conn, table):
    cur, x, gt = table["cur"], table["x"], table["gt"]
    pg = Pager()
    rng = random.Random(int(os.environ.get("TP_CORRUPT_SEED", "7")))
    n_iter = int(os.environ.get("TP_CORRUPT_ITER", "200"))
    nblocks = pg.nblocks()
    detected = errored = 0
    t0 = time.time()
    for it in range(n_iter):
        blk = rng.randrange(nblocks)
        off = rng.randrange(BLCKSZ)
        orig = pg.peek(blk, off)
        val = (orig + rng.randrange(1, 256)) & 0xFF
        pg.poke(blk, off, val)
        try:
            problems, lines = check(cur, deep=(it % 4 == 0))
            detected += problems > 0
            status, detail = owned_query(x, gt)
            errored += status == "error"
            assert alive(conn), (it, blk, off)
        finally:
            pg.poke(blk, off, orig)
    problems, lines = check(cur, deep=True)
    assert problems == 0, "\n".join(lines)
    print(f"\n{n_iter} random byte corruptions over {nblocks} pages in {time.time() - t0:.1f} s: "
          f"checker flagged {detected}, owned-path query errored {errored}, no crash")
    pg.close()


def test_extension_upgrade_from_0_2(conn, table):
    """A 0.2 install upgraded in place keeps its indexes and gains the checker (H10's upgrade script)."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_available_extension_versions WHERE name = 'trigpoint' AND version = '0.2'")
    if cur.fetchone() is None:
        pytest.skip("0.2 script not installed")
    cur.execute("DROP DATABASE IF EXISTS tp_upgrade")
    cur.execute("CREATE DATABASE tp_upgrade")
    try:
        c2 = _connect(dbname="tp_upgrade")
        k = c2.cursor()
        k.execute("CREATE EXTENSION vector")
        k.execute("CREATE EXTENSION trigpoint VERSION '0.2'")
        k.execute("CREATE TABLE t (id int, v vector(8)) WITH (autovacuum_enabled = off)")
        k.execute("INSERT INTO t SELECT i, ARRAY[random(),random(),random(),random(),random(),random(),random(),random()]::vector FROM generate_series(1, 2000) i")
        k.execute("CREATE INDEX t_ix ON t USING trigpoint (v vector_l2_ops) WITH (cells = 16, calibration_queries = 200)")
        with pytest.raises(psycopg.Error):
            k.execute("SELECT * FROM trigpoint_check('t_ix')")
        k.execute("ALTER EXTENSION trigpoint UPDATE")
        k.execute("SELECT e.extversion, a.default_version FROM pg_extension e JOIN pg_available_extensions a ON a.name = e.extname WHERE e.extname = 'trigpoint'")
        ver, default = k.fetchone()
        assert ver == default and ver >= "0.3", (ver, default)
        k.execute("SELECT * FROM trigpoint_check('t_ix', true)")
        lines = [r[0] for r in k.fetchall()]
        assert lines[-1].startswith("summary: ok"), lines
        k.execute("SET trigpoint.exact = on")
        k.execute("SELECT count(*) FROM (SELECT id FROM t ORDER BY v <-> '[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]'::vector LIMIT 10) s")
        assert k.fetchone()[0] == 10
        c2.close()
    finally:
        cur.execute("DROP DATABASE IF EXISTS tp_upgrade WITH (FORCE)")
