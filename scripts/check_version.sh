#!/bin/sh
# Checks that one version is written everywhere a release needs it: META.json (version and provides),
# pgext/trigpoint.control (default_version = major.minor), pgext/Makefile (EXTVERSION), docker/Dockerfile.release
# (VERSION), pgext/trigpoint--<major.minor>.sql and a "## <major.minor>" heading in CHANGELOG.md.
#   sh scripts/check_version.sh 0.5.0
set -e
cd "$(dirname "$0")/.."
V=${1:?usage: check_version.sh <version, e.g. 0.5.0>}
MM=$(echo "$V" | cut -d. -f1,2)
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1 = $3"; else echo "FAIL $1 = '$2', expected '$3'"; fail=1; fi; }
check "META.json version" "$(python3 -c 'import json;print(json.load(open("META.json"))["version"])')" "$V"
check "META.json provides.trigpoint.version" "$(python3 -c 'import json;print(json.load(open("META.json"))["provides"]["trigpoint"]["version"])')" "$V"
check "trigpoint.control default_version" "$(sed -n "s/^default_version *= *'\(.*\)'/\1/p" pgext/trigpoint.control)" "$MM"
check "pgext/Makefile EXTVERSION" "$(sed -n 's/^EXTVERSION *= *//p' pgext/Makefile)" "$MM"
check "Dockerfile.release VERSION" "$(sed -n 's/^ARG VERSION=//p' docker/Dockerfile.release)" "$V"
[ -f "pgext/trigpoint--$MM.sql" ] && echo "ok   pgext/trigpoint--$MM.sql" || { echo "FAIL pgext/trigpoint--$MM.sql missing"; fail=1; }
grep -qE "^## $MM( |$)" CHANGELOG.md && echo "ok   CHANGELOG.md has ## $MM" || { echo "FAIL CHANGELOG.md has no '## $MM' heading"; fail=1; }
[ $fail = 0 ] || exit 1
