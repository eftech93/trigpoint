#!/bin/sh
# The release build's own test loop: builds the test image and the release image from
# docker/Dockerfile.release for one PostgreSQL major, starts each on a throwaway server (port 5437, no volume) and runs
# the quick suites. The release image has no page hooks, so the hook-based tests skip there (conftest); the test image
# runs every suite. Works from the exported standalone tree as well as from this repository.
#
#   sh scripts/ci_release.sh            # PG_MAJOR=17
#   PG_MAJOR=18 sh scripts/ci_release.sh
#   WHICH=release sh scripts/ci_release.sh   # the release image only (test | release | both)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PG_MAJOR=${PG_MAJOR:-17}
PORT=${PORT:-5437}
WHICH=${WHICH:-both}
SUITES="tests/test_pg_extension.py tests/test_pg_check.py tests/test_pg_edge.py tests/test_pg_compact.py tests/test_pg_cosine.py tests/test_pg_security.py tests/test_pg_errcodes.py tests/test_pg_filter.py tests/test_pg_upgrade.py tests/test_pg_fuzz.py tests/test_pg_cancel.py tests/test_pg_shared_cap.py"

run_suites() {  # $1 = image, $2 = label
    name=trigpoint-release-ci
    docker rm -f $name >/dev/null 2>&1 || true
    docker run -d --rm --name $name -p $PORT:5432 --shm-size 2g -e POSTGRES_USER=vsroute -e POSTGRES_PASSWORD=vsroute -e POSTGRES_DB=vsroute \
        "$1" postgres -c shared_buffers=512MB -c maintenance_work_mem=512MB -c max_parallel_workers_per_gather=0 -c max_worker_processes=16 >/dev/null
    # pg_isready also answers for the image's temporary init server (socket only); wait for the real one over TCP
    for i in $(seq 1 60); do docker exec -e PGPASSWORD=vsroute $name psql -h 127.0.0.1 -U vsroute -d vsroute -tAc 'SELECT 1' >/dev/null 2>&1 && break; sleep 1; done
    docker exec $name psql -U vsroute -tAc "CREATE EXTENSION vector; CREATE EXTENSION trigpoint; SELECT version(); SELECT trigpoint_version();"
    PGHOST=localhost PGPORT=$PORT PGUSER=vsroute PGPASSWORD=vsroute PGDATABASE=vsroute $PY -m pytest $SUITES -q -p no:cacheprovider 2>&1 | tee /tmp/trigpoint_ci_$2.log | tail -3
    [ -f docs/operator-guide.md ] && sh scripts/doc_test.sh $name
    docker stop $name >/dev/null
    tail -1 /tmp/trigpoint_ci_$2.log | grep -q " passed" && ! tail -1 /tmp/trigpoint_ci_$2.log | grep -q "failed\|error" || { echo "$2 ($1): suites failed"; exit 1; }
}

if [ "$WHICH" = "test" ] || [ "$WHICH" = "both" ]; then
    echo "== test image (TEST_HOOKS=1), PG $PG_MAJOR"
    docker build -q -f docker/Dockerfile.release --build-arg PG_MAJOR=$PG_MAJOR --build-arg TEST_HOOKS=1 -t trigpoint-postgres-test:pg$PG_MAJOR . >/dev/null
    run_suites trigpoint-postgres-test:pg$PG_MAJOR test
fi
if [ "$WHICH" = "release" ] || [ "$WHICH" = "both" ]; then
    echo "== release image, PG $PG_MAJOR"
    docker build -q -f docker/Dockerfile.release --build-arg PG_MAJOR=$PG_MAJOR -t trigpoint/postgres:0.5-pg$PG_MAJOR . >/dev/null
    mkdir -p dist && docker build -q -f docker/Dockerfile.release --build-arg PG_MAJOR=$PG_MAJOR --target deb -o dist . >/dev/null && ls -l dist/*.deb
    run_suites trigpoint/postgres:0.5-pg$PG_MAJOR release
fi
echo "ci_release: green"
