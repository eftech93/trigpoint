-- Trigpoint 0.2 -> 0.3: the consistency checker and the test-only page poke.
\echo Use "ALTER EXTENSION trigpoint UPDATE" to load this file. \quit

-- Walks every page of the index and the flat image it holds: meta page, image chain of the current
-- generation, every log record, the TID table against the heap, the image's sections and (deep) the
-- contents of its arrays and router graph. One line per finding; the last line is
-- "summary: ... problems=N". Reads the pages, never the shared segment.
CREATE FUNCTION trigpoint_check(regclass, deep boolean DEFAULT false) RETURNS SETOF text
    AS 'MODULE_PATHNAME', 'trigpoint_check'
    LANGUAGE C STRICT VOLATILE;

-- Test hooks (superuser, trigpoint.debug = on): read / overwrite one byte of one index page through the
-- buffer manager, WAL-logged. For the corruption tests (tests/test_pg_check.py); never for data.
CREATE FUNCTION trigpoint_debug_peek(regclass, integer, integer) RETURNS integer
    AS 'MODULE_PATHNAME', 'trigpoint_debug_peek'
    LANGUAGE C STRICT VOLATILE;
CREATE FUNCTION trigpoint_debug_poke(regclass, integer, integer, integer) RETURNS void
    AS 'MODULE_PATHNAME', 'trigpoint_debug_poke'
    LANGUAGE C STRICT VOLATILE;
