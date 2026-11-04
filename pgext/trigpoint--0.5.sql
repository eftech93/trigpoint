-- Trigpoint 0.5: index access method over pgvector's vector type. "CREATE EXTENSION trigpoint".
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


-- Walks every page of the index and the flat image it holds: meta page, image chain of the current
-- generation, every log record, the TID table against the heap, the image's sections and (deep) the
-- contents of its arrays and router graph. One line per finding; the last line is
-- "summary: ... problems=N". Reads the pages, never the shared segment.
CREATE FUNCTION trigpoint_check(regclass, deep boolean DEFAULT false) RETURNS SETOF text
    AS 'MODULE_PATHNAME', 'trigpoint_check'
    LANGUAGE C STRICT VOLATILE;

-- Images pinned in shared memory across the cluster: how many (and how many still loading), their bytes,
-- and the cap trigpoint.max_shared_mb (0 = none). trigpoint_shared() lists them one per row.
CREATE FUNCTION trigpoint_shared_total()
    RETURNS TABLE(images integer, loading integer, bytes bigint, max_mb integer)
    AS 'MODULE_PATHNAME', 'trigpoint_shared_total'
    LANGUAGE C STRICT VOLATILE;

-- 0.5: whether this server can share images at all: true on PostgreSQL 17+, on 16 only
-- with shared_preload_libraries = 'trigpoint' (otherwise every backend keeps a private copy of each image it opens,
-- trigpoint_shared() is empty and trigpoint_shared_total() reports zero).
CREATE FUNCTION trigpoint_shared_available()
    RETURNS boolean
    AS 'MODULE_PATHNAME', 'trigpoint_shared_available'
    LANGUAGE C STRICT STABLE;

-- H11: cosine distance. The index stores every row normalised and ranks by L2 on the unit sphere, which is
-- the order of pgvector's <=> (1 - cosine); the executor computes the displayed distance from the heap row.
CREATE OPERATOR CLASS vector_cosine_ops FOR TYPE vector USING trigpoint AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vector, vector);

-- 0.5: the image and the router hits reveal the indexed vectors and their neighbours, so they
-- are not public; every helper that takes a regclass also requires SELECT on the indexed table (trigpoint_check(deep)
-- its ownership). The test-only page hooks of 0.3/0.4 are compiled out of release builds and no longer declared.
REVOKE EXECUTE ON FUNCTION trigpoint_image(regclass) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION trigpoint_route(regclass, vector, integer, integer) FROM PUBLIC;
