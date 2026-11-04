# Trigpoint changelog

The extension's SQL version is `trigpoint_version()`'s first field; the same call names the core revision (the hash
of `csrc/cascade.cpp` the module was built from), the SIMD kernel and the compiler. `ALTER EXTENSION trigpoint UPDATE`
moves the catalog one step at a time; the on-disk page format (3) has not changed since 0.2 and is the same on every
PostgreSQL major.

## 0.5 — 2026-11

- PostgreSQL 16, 17 and 18. On 16 the cluster-wide image registry exists only under
  `shared_preload_libraries = 'trigpoint'`; without it every backend keeps a private copy (`trigpoint_shared_available()`).
- Filtered scans: `trigpoint.iterative_scan = off | relaxed_order | strict_order` (default `relaxed_order`),
  `trigpoint.max_scan_tuples` (20,000), `trigpoint.max_probes` (0). A scan goes on in batches when the executor asks for
  more than its first k rows (a `WHERE` above the index, a `LIMIT` above `trigpoint.k`): k and the cells opened double,
  the routed order is kept, no heap row is fetched twice.
- Compaction returns its space: after the flip the new image is moved into the pages the retired generation freed and
  the relation is truncated after its last live page (`trigpoint.compact_reclaim`, on).
- Security: every SQL helper that takes a `regclass` requires `SELECT` on the indexed table (`trigpoint_check(…, true)`
  ownership); `trigpoint_image()` and `trigpoint_route()` are revoked from `PUBLIC`; `trigpoint_shared()` lists only the
  images of indexes the caller may read (all of them for superusers). The page hooks `trigpoint_debug_peek` / `_poke` and
  the result-changing diagnostics (`trigpoint.debug_skip_heap`, `trigpoint.debug_no_delta_graph`,
  `trigpoint.allow_build_threads`) exist only in test builds (`make TEST_HOOKS=1`; `trigpoint_version()` says
  `test-hooks`) and are dropped by the 0.4 -> 0.5 update.
- Error codes: every error carries a sqlstate; corruption is `XX002` (index corrupted) with a REINDEX hint, parameters
  `22023`, memory and the shared cap `53200` / `53400`.
- Robustness: a scan refuses an image value a page corruption put out of range (cell, position, graph link) instead of
  indexing with it; a compaction runs the deep checker before it reads the image; the deep checker validates the model's
  floats; `statement_timeout` and cancel stop builds, scans, compactions and checks; `trigpoint.on_shared_cap = private`
  reads a private copy at the shared-memory cap; a loader that dies leaves no half-published entry.
- Upgrade paths tested: `ALTER EXTENSION UPDATE` from 0.2 step by step, `pg_dump` round trip, `pg_upgrade` 17 -> 18,
  `CREATE INDEX` / `REINDEX INDEX CONCURRENTLY` under writers.
- `trigpoint.compact_frac` is clamped to 0.0001–100.

## 0.4 — 2026-10-28 (hardening)

- `vector_cosine_ops` (H11): the `<=>` operator class (rows and queries normalised, ranked by L2 on the unit sphere).
- `trigpoint.max_shared_mb` and `trigpoint_shared_total()` (H8): a cap on the pinned shared images and the cluster's totals.
- Compaction keeps only the live rows, renumbered (H13), repairs the router graph and re-calibrates the bound over 500
  queries (H16, `trigpoint.recalibrate_queries`, H18 `trigpoint.recalibrate_probes`); `trigpoint.compact_records` (H9)
  bounds a backend's replay; appended rows are scored flat until `trigpoint.delta_flat_max` and then linked into a
  delta graph (H17); an index over an empty table carries a trivial model (H14).
- Four latent defects found by the sanitizer build fixed: page WAL records outside critical sections, a released
  registered page, a buffer lock released after errfinish, a zero-length memcpy from NULL.

## 0.3 — 2026-10-27 (hardening H3)

- `trigpoint_check(regclass, deep)`: the consistency checker over every page and the image; the read path refuses pages
  the shallow check would flag. Test hooks `trigpoint_debug_peek` / `_poke` (dropped again in 0.5).

## 0.2 — 2026-10-26 (the spike)

- The access method: `vector_l2_ops`, a per-point int8 PCA sketch with an HNSW router over the sketches, the
  calibrated or exact lower-bound cascade, one shared image per index generation in dynamic shared memory
  (`trigpoint.shared_image`, page format 3), an append log of inserts and tombstones compacted by VACUUM, a parallel
  build on maintenance workers.
- `trigpoint_info()`, `trigpoint_image()` and `trigpoint_route()` (the parity gate against the library), `trigpoint_shared()`,
  `trigpoint_backend_memory()`, `trigpoint_version()`.
