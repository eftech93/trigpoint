"""Shared pytest setup for the extension suites.

Release builds of trigpoint.so have no page hooks and no result-changing diagnostics; test builds (make TEST_HOOKS=1,
the local loop's default) export them and report "test-hooks" in trigpoint_version(). The hooks are not part of the
extension's SQL any more, so a test build creates them here as plain superuser functions, and the hook-based suite
(test_pg_check.py) is skipped on a release build.
"""
import psycopg
import pytest

HOOK_SQL = (
    "CREATE OR REPLACE FUNCTION trigpoint_debug_peek(regclass, integer, integer) RETURNS integer "
    "AS '$libdir/trigpoint', 'trigpoint_debug_peek' LANGUAGE C STRICT VOLATILE",
    "CREATE OR REPLACE FUNCTION trigpoint_debug_poke(regclass, integer, integer, integer) RETURNS void "
    "AS '$libdir/trigpoint', 'trigpoint_debug_poke' LANGUAGE C STRICT VOLATILE",
)


def _version():
    with psycopg.connect(autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector"); c.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
        c.execute("ALTER EXTENSION trigpoint UPDATE")
        ver = c.execute("SELECT trigpoint_version()").fetchone()[0]
        # a server still on 0.4's library (before a live rebuild) has no trigpoint_shared_available(): registry present
        if c.execute("SELECT to_regprocedure('trigpoint_shared_available()') IS NOT NULL").fetchone()[0]:
            return ver, c.execute("SELECT trigpoint_shared_available()").fetchone()[0]
        return ver, True


def pytest_configure(config):
    config.addinivalue_line("markers", "shared_images: needs the cluster-wide image registry (absent on PostgreSQL 16 without shared_preload_libraries)")
    config.addinivalue_line("markers", "hooks: needs the page hooks of a test build (TEST_HOOKS=1)")
    config.tp_version, config.tp_shared_available = _version()
    config.tp_test_hooks = "test-hooks" in config.tp_version
    with psycopg.connect(autocommit=True) as c:
        if config.tp_test_hooks:
            for sql in HOOK_SQL:
                c.execute(sql)
        else:  # a release module exports no hook symbols; leftovers from a test build in the same cluster must go
            c.execute("DROP FUNCTION IF EXISTS trigpoint_debug_peek(regclass, integer, integer)")
            c.execute("DROP FUNCTION IF EXISTS trigpoint_debug_poke(regclass, integer, integer, integer)")


def pytest_collection_modifyitems(config, items):
    if not config.tp_shared_available:  # PostgreSQL 16 without shared_preload_libraries = 'trigpoint': private copies only
        skip_shared = pytest.mark.skip(reason="no shared-image registry on this server (PostgreSQL 16 without the preload)")
        for item in items:
            if "shared_images" in item.keywords:
                item.add_marker(skip_shared)
    if config.tp_test_hooks:
        return
    skip = pytest.mark.skip(reason="release build: no page hooks (TEST_HOOKS=0)")
    for item in items:
        if item.module.__name__ == "test_pg_check" or "hooks" in item.keywords:
            item.add_marker(skip)
