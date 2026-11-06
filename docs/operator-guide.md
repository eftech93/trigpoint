# Trigpoint operator guide (extension 0.5)

Trigpoint is a PostgreSQL index access method for pgvector's `vector` type. It keeps one small int8 sketch per row
(a PCA head quantised to 8 bits, plus two norms per level), routes a query through an HNSW graph over those sketches to
a few cells, prunes each cell with a lower bound on the distance before it touches a heap row, and ranks what survives
by the exact distance from the heap. The image of an index lives once in shared memory per generation; inserts and
deletes go to an append log that VACUUM folds into the next generation. `pgext/README.md` is the reference; this guide
is the order an operator meets things in.

## 1. Install

Three ways, all producing the same extension:

- **Release image**: `docker build -f docker/Dockerfile.release --build-arg PG_MAJOR=17 -t trigpoint/postgres:0.5-pg17 .`
  gives pgvector's official image plus the extension, no compiler inside. `PG_MAJOR` is 16, 17 or 18.
- **Debian package**: `docker build -f docker/Dockerfile.release --build-arg PG_MAJOR=17 --target deb -o dist .` leaves
  `dist/postgresql-17-trigpoint_0.5.0_<arch>.deb`; `dpkg -i` it on a host that has `postgresql-17` and
  `postgresql-17-pgvector` (0.8 or later).
- **From source**: `make -C pgext && make -C pgext install` with `pg_config` of the target major on the PATH; a C++17
  compiler for the core and `csrc/hnswlib` (header-only, cloned from github.com/nmslib/hnswlib).

Then, in each database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION trigpoint;             -- ALTER EXTENSION trigpoint UPDATE from an earlier version
SELECT trigpoint_version();             -- version, core revision, SIMD kernel, compiler, server
SELECT trigpoint_shared_available();    -- true on 17 and 18; on 16 only with shared_preload_libraries = 'trigpoint'
```

PostgreSQL 16 has no shared-memory registry for extensions. With `shared_preload_libraries = 'trigpoint'` in
`postgresql.conf` (a restart) the images are shared as on 17 and 18; without it the extension works unchanged but every
backend keeps its own copy of each image it opens: one image's memory per backend and a slower first query per backend.

## 2. Build an index

```sql
CREATE INDEX t_v_ix ON t USING trigpoint (v vector_l2_ops);                 -- L2, <->
CREATE INDEX t_v_cos ON t USING trigpoint (v vector_cosine_ops);            -- cosine, <=>
CREATE INDEX t_v_tuned ON t USING trigpoint (v vector_l2_ops) WITH (cells = 256, levels = '32,128,256', calibration_queries = 1000);
```

The build runs on maintenance workers (`max_parallel_maintenance_workers`) and needs `maintenance_work_mem` large
enough for every vector of the table in shared memory: about 6 bytes x rows x dimensions (n x D x 4 for the vectors
plus working arrays; 30.1 GB at 5M x 1024), and the build refuses below that, with or without workers. In a container
`dynamic_shared_memory_type = posix` needs `/dev/shm` to hold that area during the build and the image afterwards
(`shm_size`). The three options that matter:

| option | default | what it does |
|---|---|---|
| `cells` | 0 = automatic (about 75 rows per cell, a power of two from 16 to 8192) | more cells = smaller cells, fewer rows scanned per probe, a larger router |
| `levels` | by dimension (e.g. `'32,128,256'` at 1536 dims) | the cumulative widths of the sketch's levels; the last is the sketch width, the first the router's code width |
| `calibration_queries` | 2000 | corpus points used to calibrate the lower bound (0 = the exact bound only, no pruning slack) |

`seed` makes a build deterministic. A 1M x 1536 build took 286 s on an x86 server's maintenance workers;
`trigpoint_info(index)` reports the phases afterwards. `CREATE INDEX CONCURRENTLY` and `REINDEX INDEX CONCURRENTLY` work under concurrent
writers (the rows inserted meanwhile reach the log).

An index over an empty table works (a trivial model) and refits itself at the first REINDEX; create-then-load is fine,
but a REINDEX after the load gives the model the data saw.

## 3. Query

```sql
SET enable_seqscan = off;   -- only while checking that the index is used
SELECT id FROM t ORDER BY v <-> '[...]'::vector LIMIT 10;
SELECT id FROM t WHERE category = 7 ORDER BY v <-> '[...]'::vector LIMIT 10;
```

The settings that change what a scan does, all `SET`-able per session:

| setting | default | effect |
|---|---|---|
| `trigpoint.hits` / `trigpoint.probes` | 200 / 32 | router hits asked of the graph, cells opened: the recall/latency dial (more probes, more recall, latency about linear in probes once the route is paid) |
| `trigpoint.delta` | 0.01 | the calibrated bound's miss probability (the share of true neighbours the pruning may drop, in expectation) |
| `trigpoint.k` | 10 | neighbours computed by the first batch of a scan |
| `trigpoint.exact` | off | the certificate mode: exact bound over every cell, 100 % recall, several times slower |
| `trigpoint.iterative_scan` | relaxed_order | how a scan goes on when the executor asks for more than its first k rows (a filter above the index, a larger LIMIT): off = repeat with k doubled over the same cells; relaxed_order = double k and the cells opened per batch, never fetch a heap row twice; strict_order = the same, rows out of order dropped |
| `trigpoint.max_scan_tuples` / `trigpoint.max_probes` | 20000 / 0 | caps on an iterative scan (heap rows fetched, cells opened; 0 = every cell): at a cap the query returns fewer rows than asked |

Filtered queries: the index does not see the `WHERE` (no index quals). A selective filter makes the executor discard
rows the index returned; the iterative scan keeps the stream going. On the 100k-row test corpus at 10 %, 1 % and 0.1 %
selectivity a LIMIT 10 came back full with recall 1.00 / 1.00 / 0.97 against the filtered exact answer, at 4x / 18x / 60x
the unfiltered latency: a 0.1 % filter means the executor looks at about 10,000 rows for its 10. If filtered latency
matters more than recall, cap the scan (`max_scan_tuples`), and consider a partial index per category.

`EXPLAIN` shows `Index Scan using t_v_ix` with `Order By: (v <-> ...)` and the filter as `Filter:`. There is no access
method hook for EXPLAIN's output; `SET trigpoint.debug = on` prints a NOTICE per batch with the cells opened, survivors,
heap fetches and the time split (core, route, heap).

## 4. Memory

- The image of an index is about `n x (sketch width + 8 x levels + 40)` bytes plus the router graph (`n x (M0 + 1) x 4`,
  M0 = 2 x `m` = 64 by default): 0.86 KB per row at the defaults, 856 MB at 1M rows, independent of the dimension.
  `trigpoint_info(index)` reports `image_bytes` and `core_bytes`.
- One copy per generation in dynamic shared memory, mapped by every backend that scans it (`trigpoint_shared()` lists
  them, `trigpoint_shared_total()` sums them). A compaction creates the next generation; the previous one is unmapped
  when its last scan ends.
- `trigpoint.max_shared_mb` caps the pinned images cluster-wide (0 = no cap). At the cap a load fails with sqlstate
  53400, or, with `trigpoint.on_shared_cap = private`, the backend reads a private copy (one image's memory per
  backend, a LOG line each time). `trigpoint.shared_image = off` makes every backend read a private copy always.
- `trigpoint_backend_memory()` is this backend's resident size; a private copy shows up there.
- The container's `/dev/shm` (compose `shm_size`) must hold every shared image plus, during a build, the build's vector area.
  A dropped index's image is released by the next build or image load in its database, so a drop followed by a rebuild
  needs room for the build's area, not the old image too, unless a session that used the old index has not run a
  statement since the drop (it still maps the image until it does).

## 5. Maintenance

- **Inserts** append to the index's log: a record per row (the sketch and norms, no vector), ~4 ms per row at 1M.
  A backend replays the log since the image at its first scan after a change (`trigpoint.compact_records`, 20,000,
  bounds that replay by forcing a compaction).
- **Deletes** are tombstones in the log; the executor never sees a deleted row.
- **VACUUM** compacts when inserts plus tombstones since the image exceed `trigpoint.compact_frac` (0.2) of its rows,
  or when the log holds more than `trigpoint.compact_records` records: the live rows are rewritten into a new
  generation, the router repaired, the bound re-calibrated (`trigpoint.recalibrate_queries`, 500), and the relation
  truncated to the live image plus one page (`trigpoint.compact_reclaim`, on). Cost by size on an x86 server:
  about two minutes at 1M x 1536 (123–129 s after deleting 10 % and 25 %), of which 20–40 s is the re-calibration.
- **REINDEX** rebuilds from scratch (the model, cells and calibration refit to the current rows): after a large change
  in the data's distribution, or after corruption.
- **Upgrades**: `ALTER EXTENSION trigpoint UPDATE` changes the catalog only, one version step at a time from 0.2. The
  page format (3) is the same across releases and PostgreSQL majors; `pg_upgrade` carries indexes across majors without
  a REINDEX; `pg_dump` stores the CREATE INDEX and the restore rebuilds.

## 6. Monitoring and checking

```sql
SELECT trigpoint_info('t_v_ix');                 -- rows, cells, levels, image size, generation, log counts, compactions, build phases
SELECT * FROM trigpoint_shared();                -- the shared images: index, generation, bytes, loading, loaded at
SELECT * FROM trigpoint_shared_total();          -- images, loading, bytes, the cap
SELECT * FROM trigpoint_check('t_v_ix', false);  -- pages, image header and sections, log records (fast)
SELECT * FROM trigpoint_check('t_v_ix', true);   -- also every array's contents and the model's floats (O(n), ~1.6 µs per row)
```

The checker's last row is `summary: ok|CORRUPT, N problems`. Run the deep form after a suspicious crash, before a
pg_upgrade, or when a query raised sqlstate `XX002`.

## 7. Failure modes

| symptom | meaning | action |
|---|---|---|
| `ERROR ... (sqlstate XX002)`, "corrupt image", "page format", "index image truncated" | a page of the index does not agree with itself; the reader refused it rather than use it | `trigpoint_check(index, true)` names the page; `REINDEX INDEX` rebuilds |
| wrong neighbours with no error, checker clean | a flipped byte in the model's floats that no structural check can see | `REINDEX`; enable data checksums (`initdb --data-checksums`, the default from PostgreSQL 18) so the page itself is refused |
| `would exceed trigpoint.max_shared_mb` (53400) | the cap on shared images | raise the cap, drop unused trigpoint indexes, or `trigpoint.on_shared_cap = private` |
| `no free slot in the shared image table` (53000) | more live image generations than the registry's slots | drop unused indexes; the registry sweeps dropped ones itself |
| the first query of a backend is slow after a VACUUM | the compaction wrote a new generation; the first backend to need it loads it into shared memory (seconds at 1M), the others attach | expected; `trigpoint.debug = on` shows the load line |
| a query returns fewer rows than LIMIT under a filter | the iterative scan reached `trigpoint.max_scan_tuples` or `trigpoint.max_probes` | raise the cap for that session |
| `query has D dimensions, index has D'` | the query's dimension differs from the column's | fix the query |
| a build or VACUUM cancelled by `statement_timeout` | the statement stopped at its next interrupt check | the index is left as it was (no index, or the old generation); rerun without the timeout |

## 8. Limits

- `vector` columns up to pgvector's dimension limit; one indexed column per index; no `INCLUDE` columns; no index quals
  (a `WHERE` on the indexed column itself is refused); no backward scans; no planner-parallel scans.
- L2 and cosine; inner product (`<#>`) and `halfvec` are not supported in 0.5.
- Linux x86-64 (AVX2) and arm64 (NEON) are tested; the core compiles with scalar kernels elsewhere.
- The calibrated bound's guarantee is statistical and assumes queries like the calibration set; a strong drift in the
  queries lowers recall until a REINDEX (or a compaction, which re-calibrates over the current rows).
