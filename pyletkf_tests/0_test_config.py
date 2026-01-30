from __future__ import annotations

import re
from pathlib import Path

import pytest
from pyletkf import params as py_params

CONFIG_PATH = Path("test/SC23/conf/letkf_20210730060030.conf")


def _parse_value(pattern: str) -> float:
    text = CONFIG_PATH.read_text()
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"Pattern '{pattern}' not found in {CONFIG_PATH}")
    token = match.group(1).strip().rstrip(",")
    if token.lower() in {"true", ".true."}:
        return 1.0
    if token.lower() in {"false", ".false."}:
        return 0.0
    return float(token.replace("D", "E"))


def test_parameters_match_configuration():
    assert len(py_params.MEMBERS) == int(
        _parse_value(r"^\s*MEMBER\s*=\s*([0-9]+)")
    )
    assert py_params.HORI_LOCAL_RADAR_OBSNOREF == pytest.approx(
        _parse_value(r"HORI_LOCAL_RADAR_OBSNOREF\s*=\s*([0-9.DdEe+-]+)")
    )
    assert py_params.MAX_OBS_PER_GRID == int(
        _parse_value(r"MAX_NOBS_PER_GRID\s*=\s*([0-9]+)")
    )
    assert py_params.RELAX_ALPHA == pytest.approx(
        _parse_value(r"RELAX_ALPHA\s*=\s*([0-9.DdEe+-]+)")
    )
    assert py_params.Q_SPRD_MAX == pytest.approx(
        _parse_value(r"Q_SPRD_MAX\s*=\s*([0-9.DdEe+-]+)")
    )
