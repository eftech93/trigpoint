-- Trigpoint 0.4 -> 0.5 (security).
\echo Use "ALTER EXTENSION trigpoint UPDATE" to load this file. \quit

-- The test-only page hooks are compiled out of release builds; drop their SQL entries so no function points at a
-- symbol the module no longer exports.
DROP FUNCTION IF EXISTS trigpoint_debug_peek(regclass, integer, integer);
DROP FUNCTION IF EXISTS trigpoint_debug_poke(regclass, integer, integer, integer);

-- The image and the router hits reveal the indexed vectors and their neighbours: not public. (Every regclass helper
-- now also checks SELECT on the indexed table in C.)
REVOKE EXECUTE ON FUNCTION trigpoint_image(regclass) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION trigpoint_route(regclass, vector, integer, integer) FROM PUBLIC;

-- 0.5: whether this server can share images at all: true on PostgreSQL 17+, on 16 only
-- with shared_preload_libraries = 'trigpoint' (otherwise every backend keeps a private copy of each image it opens,
-- trigpoint_shared() is empty and trigpoint_shared_total() reports zero).
CREATE FUNCTION trigpoint_shared_available()
    RETURNS boolean
    AS 'MODULE_PATHNAME', 'trigpoint_shared_available'
    LANGUAGE C STRICT STABLE;
