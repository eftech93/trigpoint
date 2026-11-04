-- Trigpoint 0.3 -> 0.4 (hardening H8): the cluster-wide totals of the shared images.
\echo Use "ALTER EXTENSION trigpoint UPDATE" to load this file. \quit

-- Images pinned in shared memory across the cluster: how many (and how many still loading), their bytes,
-- and the cap trigpoint.max_shared_mb (0 = none). trigpoint_shared() lists them one per row.
CREATE FUNCTION trigpoint_shared_total()
    RETURNS TABLE(images integer, loading integer, bytes bigint, max_mb integer)
    AS 'MODULE_PATHNAME', 'trigpoint_shared_total'
    LANGUAGE C STRICT VOLATILE;

-- H11: cosine distance. The index stores every row normalised and ranks by L2 on the unit sphere, which is
-- the order of pgvector's <=> (1 - cosine); the executor computes the displayed distance from the heap row.
CREATE OPERATOR CLASS vector_cosine_ops FOR TYPE vector USING trigpoint AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vector, vector);
