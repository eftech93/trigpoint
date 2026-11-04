#!/bin/sh
# The operator guide's SQL runs as written. Every ```sql block of docs/operator-guide.md is run
# through psql (ON_ERROR_STOP) in a scratch database of the given container, against a table t(id, category, v
# vector(1536)) of 4,000 random rows; '[...]' stands for a real vector literal. Exit 1 on the first error.
#   sh scripts/doc_test.sh trigpoint-postgres-local
set -e
cd "$(dirname "$0")/.."
C=${1:-trigpoint-postgres-local}
PSQL="docker exec -i $C psql -U vsroute -v ON_ERROR_STOP=1 -q"
$PSQL -d postgres -c "DROP DATABASE IF EXISTS tp_doc" -c "CREATE DATABASE tp_doc" >/dev/null
$PSQL -d tp_doc <<'SQL' >/dev/null
CREATE EXTENSION vector;
SELECT setseed(0.1);
CREATE TABLE t (id int PRIMARY KEY, category int, v vector(1536));
INSERT INTO t SELECT i, i % 20, (SELECT array_agg(random()::real) FROM generate_series(1, 1536))::vector FROM generate_series(1, 4000) i;
SQL
LIT=$(docker exec $C psql -U vsroute -tA -d tp_doc -c "SELECT v::text FROM t WHERE id = 1")
# every ```sql block, '[...]' replaced by the literal, the first CREATE EXTENSION made idempotent (the table above needed vector)
${PY:-python3} - docs/operator-guide.md "$LIT" > /tmp/tp_doc_blocks.sql <<'PY'
import sys
text, lit = open(sys.argv[1]).read(), sys.argv[2]
on = False
for line in text.splitlines():
    if line.startswith("```sql"):
        on = True; continue
    if line.startswith("```"):
        if on: print("-- block end")
        on = False; continue
    if on:
        print(line.replace("'[...]'::vector", "'%s'::vector" % lit).replace("CREATE EXTENSION vector;", "CREATE EXTENSION IF NOT EXISTS vector;"))
PY
n=$(grep -c "^-- block end" /tmp/tp_doc_blocks.sql)
$PSQL -d tp_doc < /tmp/tp_doc_blocks.sql > /tmp/tp_doc_out.txt 2>&1 || { tail -5 /tmp/tp_doc_out.txt; echo "doc_test: a guide block failed"; exit 1; }
$PSQL -d postgres -c "DROP DATABASE tp_doc" >/dev/null
echo "doc_test: $n SQL blocks of docs/operator-guide.md ran clean"
