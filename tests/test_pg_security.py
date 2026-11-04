"""The helpers that take a regclass check the caller's SELECT privilege on the indexed table
(ownership for trigpoint_check(deep)); trigpoint_image and trigpoint_route are revoked from PUBLIC; a release build
has no page hooks and no result-changing diagnostic GUCs.

    PGHOST=... PGUSER=<superuser> pytest tests/test_pg_security.py -q
"""
import os

import numpy as np
import psycopg
import pytest

D, N = 32, 3000
ROLE, PW = "tp_sec_reader", "tp_sec_pw"


def lit(v):
    return "[" + ",".join(repr(float(t)) for t in v) + "]"


@pytest.fixture(scope="module")
def setup():
    su = psycopg.connect(autocommit=True)
    cur = su.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector"); cur.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
    cur.execute(f"DROP ROLE IF EXISTS {ROLE}")
    cur.execute(f"CREATE ROLE {ROLE} LOGIN PASSWORD '{PW}'")
    cur.execute("DROP TABLE IF EXISTS tp_sec")
    cur.execute(f"CREATE TABLE tp_sec (id int PRIMARY KEY, v vector({D}))")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(N, D)).astype(np.float32)
    with cur.copy("COPY tp_sec (id, v) FROM STDIN") as cp:
        for i, row in enumerate(x):
            cp.write_row((i, lit(row)))
    cur.execute("CREATE INDEX tp_sec_ix ON tp_sec USING trigpoint (v vector_l2_ops) WITH (cells = 16, calibration_queries = 50)")
    cur.execute("SELECT id FROM tp_sec ORDER BY v <-> %s::vector LIMIT 5", (lit(x[0]),))  # loads the shared image
    yield su, x
    cur.execute("DROP TABLE IF EXISTS tp_sec")
    cur.execute(f"REASSIGN OWNED BY {ROLE} TO CURRENT_USER"); cur.execute(f"DROP OWNED BY {ROLE}"); cur.execute(f"DROP ROLE IF EXISTS {ROLE}")
    su.close()


def reader():
    return psycopg.connect(user=ROLE, password=PW, dbname=os.environ.get("PGDATABASE", "vsroute"), autocommit=True)


def denied(conn, sql, params=None):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute(sql, params).fetchall()


def test_without_select_everything_is_denied(setup):
    _, x = setup
    with reader() as r:
        denied(r, "SELECT trigpoint_info('tp_sec_ix')")
        denied(r, "SELECT trigpoint_image('tp_sec_ix')")
        denied(r, "SELECT * FROM trigpoint_route('tp_sec_ix', %s::vector, 10, 50)", (lit(x[0]),))
        denied(r, "SELECT * FROM trigpoint_check('tp_sec_ix', false)")
        denied(r, "SELECT * FROM trigpoint_check('tp_sec_ix', true)")
        assert r.execute("SELECT count(*) FROM trigpoint_shared() WHERE indexrelid = 'tp_sec_ix'::regclass").fetchone()[0] == 0


@pytest.mark.shared_images
def test_select_grants_the_read_only_helpers_but_not_the_data(setup):
    su, x = setup
    su.execute(f"GRANT SELECT ON tp_sec TO {ROLE}")
    try:
        with reader() as r:
            assert r.execute("SELECT trigpoint_info('tp_sec_ix')").fetchone()[0].startswith(f"n={N} ")
            assert r.execute("SELECT * FROM trigpoint_check('tp_sec_ix', false)").fetchall()[-1][0].startswith("summary: ok")
            assert r.execute("SELECT count(*) FROM trigpoint_shared() WHERE indexrelid = 'tp_sec_ix'::regclass").fetchone()[0] == 1
            denied(r, "SELECT * FROM trigpoint_check('tp_sec_ix', true)")          # deep: owner only
            denied(r, "SELECT trigpoint_image('tp_sec_ix')")                      # revoked from PUBLIC
            denied(r, "SELECT * FROM trigpoint_route('tp_sec_ix', %s::vector, 10, 50)", (lit(x[0]),))
            # a plain k-NN query through the index works like any SELECT
            assert len(r.execute("SELECT id FROM tp_sec ORDER BY v <-> %s::vector LIMIT 5", (lit(x[1]),)).fetchall()) == 5
    finally:
        su.execute(f"REVOKE SELECT ON tp_sec FROM {ROLE}")


def test_select_without_ownership_is_refused_the_deep_check_owner_is_not_and_revoked_functions_are_grantable(setup):
    su, x = setup
    assert su.execute("SELECT * FROM trigpoint_check('tp_sec_ix', true)").fetchall()[-1][0].startswith("summary: ok")
    su.execute(f"GRANT SELECT ON tp_sec TO {ROLE}"); su.execute(f"GRANT EXECUTE ON FUNCTION trigpoint_image(regclass) TO {ROLE}")
    try:
        with reader() as r:
            assert len(r.execute("SELECT trigpoint_image('tp_sec_ix')").fetchone()[0]) > 0
    finally:
        su.execute(f"REVOKE EXECUTE ON FUNCTION trigpoint_image(regclass) FROM {ROLE}"); su.execute(f"REVOKE SELECT ON tp_sec FROM {ROLE}")


def test_not_an_index_is_rejected_cleanly(setup):
    su, _ = setup
    with pytest.raises(psycopg.errors.WrongObjectType):
        su.execute("SELECT trigpoint_info('tp_sec')").fetchall()
    with pytest.raises(psycopg.errors.WrongObjectType):
        su.execute("SELECT trigpoint_image('tp_sec_pkey')").fetchall()


def test_release_build_has_no_hooks_or_diagnostics(setup, request):
    su, _ = setup
    version = su.execute("SELECT trigpoint_version()").fetchone()[0]
    ext_funcs = {r[0] for r in su.execute("SELECT p.proname FROM pg_depend d JOIN pg_proc p ON p.oid = d.objid JOIN pg_extension e ON e.oid = d.refobjid WHERE e.extname = 'trigpoint'").fetchall()}
    assert not {f for f in ext_funcs if f.startswith("trigpoint_debug_")}, "the extension's SQL must not declare the hooks"
    if "test-hooks" in version:
        pytest.skip("test build: hooks and diagnostics present by design")
    for guc in ("trigpoint.debug_skip_heap", "trigpoint.debug_no_delta_graph", "trigpoint.allow_build_threads"):
        with pytest.raises(psycopg.errors.UndefinedObject):
            su.execute(f"SHOW {guc}").fetchall()
    assert su.execute("SELECT count(*) FROM pg_proc WHERE proname LIKE 'trigpoint_debug_%'").fetchone()[0] == 0
