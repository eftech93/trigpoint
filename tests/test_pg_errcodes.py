"""Every ERROR the extension raises carries an SQLSTATE (errcode) — checked on the sources,
which the test container sees under /work — and the user-reachable ones surface with the expected class."""
import pathlib
import re

import psycopg
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "pgext"


def test_every_error_has_an_errcode():
    missing = []
    for f in sorted(SRC.glob("*.c")):
        src = f.read_text()
        for m in re.finditer(r"ereport\(\s*ERROR\s*,\s*\((.*?)\)\)\s*;", src, re.S):
            if "errcode(" not in m.group(1):
                missing.append(f"{f.name}:{src[:m.start()].count(chr(10)) + 1}")
    assert not missing, missing


def test_user_errors_have_the_right_class():
    with psycopg.connect(autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector"); c.execute("CREATE EXTENSION IF NOT EXISTS trigpoint")
        c.execute("DROP TABLE IF EXISTS tp_err"); c.execute("CREATE TABLE tp_err (id int PRIMARY KEY, v vector)")
        with pytest.raises(psycopg.errors.InvalidParameterValue):   # typmod-less column
            c.execute("CREATE INDEX tp_err_ix ON tp_err USING trigpoint (v vector_l2_ops)")
        c.execute("DROP TABLE tp_err"); c.execute("CREATE TABLE tp_err (id int PRIMARY KEY, v vector(8))")
        with pytest.raises(psycopg.errors.InvalidParameterValue):   # levels above the dimension
            c.execute("CREATE INDEX tp_err_ix ON tp_err USING trigpoint (v vector_l2_ops) WITH (levels = '4,16')")
        with pytest.raises(psycopg.errors.InvalidParameterValue):
            c.execute("SET trigpoint.delta = 2")
        c.execute("DROP TABLE tp_err")
