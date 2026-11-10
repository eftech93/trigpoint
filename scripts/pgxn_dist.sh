#!/bin/sh
# The PGXN source distribution: dist/trigpoint-<version>.zip, made from a committed tree (git archive of HEAD) plus
# hnswlib's headers at the pinned commit (header-only, Apache-2.0; its licence goes along), so `pgxn install trigpoint`
# builds without network access. Upload the zip at https://manager.pgxn.org.
#   sh scripts/pgxn_dist.sh            # version from META.json
set -e
cd "$(dirname "$0")/.."
V=${1:-$(python3 -c 'import json;print(json.load(open("META.json"))["version"])')}
HNSWLIB_REF=${HNSWLIB_REF:-$(sed -n 's/^ARG HNSWLIB_REF=//p' docker/Dockerfile.release)}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git archive --format=tar --prefix=trigpoint-$V/ HEAD | tar -x -C "$T"
rm -rf "$T/trigpoint-$V/.github"
git clone -q https://github.com/nmslib/hnswlib "$T/hnswlib-src" && git -C "$T/hnswlib-src" checkout -q "$HNSWLIB_REF"
mkdir -p "$T/trigpoint-$V/csrc/hnswlib/hnswlib"
cp "$T/hnswlib-src/hnswlib/"*.h "$T/trigpoint-$V/csrc/hnswlib/hnswlib/"
cp "$T/hnswlib-src/LICENSE" "$T/trigpoint-$V/csrc/hnswlib/LICENSE"
echo "hnswlib $HNSWLIB_REF (https://github.com/nmslib/hnswlib), headers only" > "$T/trigpoint-$V/csrc/hnswlib/VERSION"
mkdir -p dist; rm -f "dist/trigpoint-$V.zip"
OUT=$(pwd)/dist/trigpoint-$V.zip
(cd "$T" && python3 -m zipfile -c "$OUT" "trigpoint-$V/")
python3 -m zipfile -l "$OUT" | grep -c . | xargs echo "dist/trigpoint-$V.zip: entries (incl. header):"
