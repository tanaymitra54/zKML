"""Regression test for the reborn bare/bound binding harness (paper Table 2).

The paper's central claim is that the binding layer costs a constant 899
constraint rows. This test pins the measurement, not the prose: if a future edit
to ``zk.binding`` changes the circuit shape, the delta moves and this fails.
"""

from __future__ import annotations

import csv
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "results" / "zk_binding.csv"

SYNTHETIC_BARE_ROWS = {
    "syn_8x4": 1356,
    "syn_16x4": 2524,
    "syn_32x4": 4860,
    "syn_16x8": 5048,
    "syn_32x8": 9720,
}
BINDING_ROWS = 899


def _rows() -> dict[str, dict[str, str]]:
    with CSV.open(newline="") as fh:
        return {row["config"]: row for row in csv.DictReader(fh)}


def test_bare_rows_match_the_paper() -> None:
    rows = _rows()
    for config, expected in SYNTHETIC_BARE_ROWS.items():
        assert config in rows, f"{config} missing from {CSV.name}"
        assert int(rows[config]["rows_bare"]) == expected


def test_binding_delta_is_constant_across_geometries_and_scale() -> None:
    rows = _rows()
    assert len(rows) == len(SYNTHETIC_BARE_ROWS) + 1
    for config, row in rows.items():
        assert int(row["delta_rows"]) == BINDING_ROWS, config
        assert int(row["rows_bound"]) == int(row["rows_bare"]) + BINDING_ROWS, config


def test_large_circuit_control_is_present() -> None:
    row = _rows()["mlp_h384"]
    assert int(row["rows_bare"]) == 61_048
    assert int(row["rows_bound"]) == 61_947
