# Trigpoint — operator notes (extension 0.5)

A PostgreSQL 16, 17 or 18 index access method for pgvector's `vector` type: a per-point int8 PCA sketch with an
HNSW router over the sketches, a calibrated or exact lower bound that prunes before heap fetches, one
shared in-memory image per index, and a log of inserts and tombstones compacted by VACUUM. L2 (`<->`)
and cosine (`<=>`) operator classes. Version 0.5; `CHANGELOG.md` lists the releases.

## Install

Trigpoint needs PostgreSQL 16, 17 or 18 and pgvector 0.8 or later. Releases: https://github.com/eftech93/trigpoint/releases
and https://pgxn.org/dist/trigpoint/.

- **PGXN** (builds from source: `pg_config` of the target server on the PATH and a C++17 compiler; the build is tuned
  for the machine's CPU):

  ```
  pip install pgxnclient
  pgxn install vector          # pgvector, if it is not installed yet
  pgxn install trigpoint
  ```

- **Debian/Ubuntu package** from the GitHub release, for the PGDG packages of your major (x86-64 with AVX2 and FMA,
  i.e. Haswell or later):

  ```
  wget https://github.com/eftech93/trigpoint/releases/download/v0.5.0/postgresql-17-trigpoint_0.5.0_amd64.deb
  sudo apt install ./postgresql-17-trigpoint_0.5.0_amd64.deb      # needs postgresql-17 and postgresql-17-pgvector
  ```

- **Docker**: pgvector's official image plus the extension, no compiler inside (`PG_MAJOR` 16, 17 or 18); the same
  Dockerfile also produces the .deb:

  ```
  docker build -f docker/Dockerfile.release --build-arg PG_MAJOR=17 -t trigpoint/postgres:0.5-pg17 .
  docker build -f docker/Dockerfile.release --build-arg PG_MAJOR=17 --target deb -o dist .     # dist/*.deb
  ```

- **From a git checkout**: hnswlib (header-only) goes into `csrc/hnswlib` at the commit the release was tested with, then
  the top-level Makefile builds and installs the extension (`ARCHFLAGS=-march=x86-64-v3` for a build that runs on
  other x86-64 machines; the default is `-march=native`):

  ```
  git clone https://github.com/nmslib/hnswlib csrc/hnswlib
  git -C csrc/hnswlib checkout ca426729609b3221563047ceb269cc1213e0d376
  make && sudo make install
  ```

Then, in each database:

```
CREATE EXTENSION vector;
CREATE EXTENSION trigpoint;          -- or ALTER EXTENSION trigpoint UPDATE from 0.2 / 0.3 / 0.4
CREATE INDEX t_v_l2 ON t USING trigpoint (v vector_l2_ops) WITH (cells = 0);      -- L2, <->; cells = 0: automatic
CREATE INDEX t_v_cos ON t USING trigpoint (v vector_cosine_ops);                    -- or cosine, <=>
```

The build runs on maintenance workers (`max_parallel_maintenance_workers`) and needs `maintenance_work_mem`
large enough for the vectors in shared memory: about 6 bytes x rows x dimensions (30.1 GB at 5M x 1024), or the
build refuses. With `dynamic_shared_memory_type = posix` in a container, `/dev/shm` must hold that area during the
build and the image afterwards (compose: `shm_size`). An index over an empty table works (a trivial model)
and refits itself at the first REINDEX; TRUNCATE and create-then-load are fine.

PostgreSQL versions: 17 and 18 need nothing else. On 16 the cluster-wide image registry (the shared images of
`trigpoint.shared_image`) exists only with `shared_preload_libraries = 'trigpoint'`; without the preload the extension
works unchanged but every backend keeps a private copy of each image it opens (one image's worth of memory per
backend, slower first queries), `trigpoint_shared()` is empty and `trigpoint_shared_available()` returns false.
Build the image for a major with `PG_MAJOR=16|17|18` (`docker/Dockerfile.release`); `PG_MAJOR=16 sh scripts/ci_release.sh`
runs the quick suites on that major.

Reloptions: `cells` (0 = automatic), `levels` (e.g. `'32,96,256'`), `m` and `ef_construction` (router graph),
`sample`, `kmeans_iters`, `calibration_queries`, `seed`.

## Query-time GUCs

| GUC | default | meaning |
|---|---|---|
| `trigpoint.k` | 10 | neighbours computed by the first batch of a scan; a larger LIMIT, or a `WHERE` above the index, goes on in batches |
| `trigpoint.iterative_scan` | relaxed_order | how a scan goes on when the executor asks for more rows than the first batch: `off` repeats the search with k doubled over the same cells (0.4's behaviour); `relaxed_order` doubles k and the cells opened per batch, keeps the routed cell order, cascades the opened cells again under the larger tau and never fetches a heap row twice, so a later batch may return a row closer than one already returned; `strict_order` drops such rows, so the stream stays in distance order (with `trigpoint.exact = on` every prefix is the exact top-N) |
| `trigpoint.max_scan_tuples` / `trigpoint.max_probes` | 20000 / 0 | caps on an iterative scan: heap rows fetched in all, cells opened in all (0 = every cell); at a cap the scan stops and the query returns fewer rows than asked, with no error |
| `trigpoint.hits` / `trigpoint.probes` / `trigpoint.ef` | 200 / 32 / hits | router hits, cells opened, beam width |
| `trigpoint.delta` | 0.01 | miss probability of the calibrated bound |
| `trigpoint.exact` | off | certificate mode: exact bound over every cell (100 % recall, slower) |
| `trigpoint.shared_image` | on | share one image per index in dynamic shared memory; off = a private copy per backend |
| `trigpoint.on_shared_cap` | error | at `trigpoint.max_shared_mb`: `error` fails the query (53400), `private` reads a private copy in this backend instead (one image's memory per backend; a LOG line each time) |

## Maintenance GUCs

| GUC | default | meaning |
|---|---|---|
| `trigpoint.compact_frac` | 0.2 | VACUUM rewrites the image when log inserts + tombstones exceed this share of the live rows |
| `trigpoint.compact_reclaim` | on | after a compaction, move the new image into the pages the retired generation freed and truncate the relation after its last live page (off: the file never shrinks) |
| `trigpoint.compact_records` | 20000 | … or when the log holds more records than this (bounds a backend's replay at its first scan) |
| `trigpoint.delta_flat_max` | 50000 | appended positions a backend's router scores flat before it links them into a delta graph (0: link each one as it arrives) |
| `trigpoint.recalibrate_probes` | 64 | cells each calibration query of a compaction opens, in the router's order (0: every cell) |
| `trigpoint.recalibrate_queries` | 500 | calibration queries a compaction runs over the live rows (0 = keep the build's tables) |
| `trigpoint.max_shared_mb` | 0 | superuser; cap on the pinned images of the cluster, 0 = none; a load beyond it errors |
| `trigpoint.debug` | off | per-query statistics as NOTICEs; also enables the test-only page hooks |

## Monitoring and checking

- `trigpoint_info(regclass)`: the meta page (rows, metric, cells, levels, generation, log counters, build timings).
- `trigpoint_shared()`: one row per image in shared memory; `trigpoint_shared_total()`: images, loading, bytes, cap.
- `trigpoint_check(regclass, deep boolean)`: walks every page and the image; the last row is `summary: ok|CORRUPT, N problems`.
  A reader refuses anything the shallow check would flag, and a scan refuses a value a page corruption put out of range
  (a cell, a position, a graph link) with sqlstate XX002 instead of using it; the deep check also validates the arrays'
  contents and the model's floats (PCA rows, centroids, norms), and a compaction runs it before it reads the image.
  A corrupted model float the deep check does not catch gives a wrong answer without an error: PostgreSQL's data
  checksums (`initdb --data-checksums`, the default from 18) are the protection against bit rot in pages.
  `tests/test_pg_fuzz.py` mutates random bytes (1,000 per run) and asserts XX002 or an answer, never a crash. `REINDEX` rebuilds.
- `trigpoint_backend_memory()`: this backend's resident memory.

## Operations

- VACUUM tombstones deleted rows and, above the thresholds, rewrites the image over the live rows only
  (renumbered, router graph repaired, bound re-calibrated); the relation holds at most two generations plus the log.
- Crash recovery, UNLOGGED tables and hot standbys are tested (`tests/pg_crash.py`, `tests/pg_standby.py`); `statement_timeout`
  and `pg_cancel_backend` stop a build, an exact scan, a compaction VACUUM and a deep check within the timeout plus the
  current cell or page (`tests/test_pg_cancel.py`); a backend terminated while it loads a shared image publishes its
  whole entry or leaves none (`tests/test_pg_shared_cap.py`).
- Dumps: `pg_dump` stores the CREATE INDEX statement; the image is rebuilt on restore (the round trip
  was tested: counts, validity, the checker, identical exact answers).
- On-disk format: page format 3 is read by every 0.x and 1.x release and is the same on every PostgreSQL major (fixed-width
  page structs, no PostgreSQL-version-dependent layout), so `pg_upgrade` carries a trigpoint index across majors without a
  REINDEX (tested from 17 to 18: same pages, checker clean,
  same answers). A reader refuses any other format with a REINDEX hint; a future bump ships an in-place migration or a
  documented REINDEX. `ALTER EXTENSION trigpoint UPDATE` changes the catalog only, one step at a time from 0.2
  (`tests/test_pg_upgrade.py`).
- `CREATE INDEX CONCURRENTLY` and `REINDEX INDEX CONCURRENTLY` work under concurrent inserts and deletes (the validate
  phase hands the rows inserted meanwhile to the log; `tests/test_pg_upgrade.py`).
- Filtered queries (`WHERE` on other columns with `ORDER BY v <-> q LIMIT k`): the filter is applied above the index,
  and the scan goes on in batches until the executor is satisfied (`trigpoint.iterative_scan`). On the local loop's
  100k-row corpus, at 10 %, 1 % and 0.1 % selectivity, LIMIT 10 comes back full with recall@10 1.00 / 1.00 / 0.97
  against the filtered exact answer, at 4x / 18x / 60x the unfiltered scan's 0.4 ms (the executor has to look at
  about 10 / s rows); a few 0.1 % queries hit the 20,000-row cap and come back short, `trigpoint.max_scan_tuples` raises
  it. A `WHERE` on the indexed column itself is refused (no index quals).
- Not supported: inner product (`<#>`), backward scans, `INCLUDE` columns, planner-parallel scans; one indexed column
  per index.

## Tests

`sh scripts/ci_release.sh` builds the test image (page hooks compiled in) and the release image for one major
(`PG_MAJOR=16|17|18`, `WHICH=test|release|both`) and runs the quick suites (`tests/test_pg_*.py`) against a throwaway
server; it needs Docker and a venv with psycopg, numpy and pytest. `tests/pg_soak.py`, `tests/pg_crash.py` and
`tests/pg_standby.py` are the longer soak, crash-recovery and standby runs. CI runs the suites on every major for both
images; a tag `vX.Y.Z` builds and tests the packages and drafts the GitHub release.
