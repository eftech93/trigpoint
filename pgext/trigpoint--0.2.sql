-- Trigpoint 0.2 (spike S2, E1): index access method over pgvector's vector type. "CREATE EXTENSION trigpoint".
\echo Use "CREATE EXTENSION trigpoint" to load this file. \quit

-- Extension version, core revision, distance kernel the core was compiled with, compiler flags,
-- server version — the smoke tests call this.
CREATE FUNCTION trigpoint_version() RETURNS text
    AS 'MODULE_PATHNAME', 'trigpoint_version'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- What the meta page records about an index: rows, cells, levels, pages, build timings.
CREATE FUNCTION trigpoint_info(regclass) RETURNS text
    AS 'MODULE_PATHNAME', 'trigpoint_info'
    LANGUAGE C STRICT;

CREATE FUNCTION trigpoint_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'trigpoint_handler'
    LANGUAGE C;

CREATE ACCESS METHOD trigpoint TYPE INDEX HANDLER trigpoint_handler;
COMMENT ON ACCESS METHOD trigpoint IS 'Trigpoint: per-point sketch router + calibrated int8 cascade (L2)';

-- Same shape as pgvector's hnsw/ivfflat L2 class so existing "ORDER BY v <-> q LIMIT k" queries use it.
CREATE OPERATOR CLASS vector_l2_ops DEFAULT FOR TYPE vector USING trigpoint AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_squared_distance(vector, vector);

-- 0.2 (E1, shared image): the images this cluster holds in shared memory — one row per
-- (database, relfilenode, generation) — and this backend's resident memory as the kernel reports it
-- (VmRSS, RssAnon, RssFile, RssShmem, VmHWM).
CREATE FUNCTION trigpoint_shared()
    RETURNS TABLE(dbid oid, relfilenode oid, indexrelid regclass, generation bigint, bytes bigint,
                  loading boolean, loaded_at timestamptz, load_ms float8)
    AS 'MODULE_PATHNAME', 'trigpoint_shared'
    LANGUAGE C STRICT;

CREATE FUNCTION trigpoint_backend_memory() RETURNS text
    AS 'MODULE_PATHNAME', 'trigpoint_backend_memory'
    LANGUAGE C STRICT VOLATILE;

-- E2: the index image as the pages hold it (TID table + flat image) and the router's hits for a query —
-- for the graph parity gate and routing experiments.
CREATE FUNCTION trigpoint_image(regclass) RETURNS bytea
    AS 'MODULE_PATHNAME', 'trigpoint_image'
    LANGUAGE C STRICT;

CREATE FUNCTION trigpoint_route(regclass, vector, integer, integer)
    RETURNS TABLE(tid tid, score real)
    AS 'MODULE_PATHNAME', 'trigpoint_route'
    LANGUAGE C STRICT;
