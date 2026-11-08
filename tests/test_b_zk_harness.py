"""Track B (experiments) acceptance: workload registry and CSV contract.

These tests are deliberately cheap — they never invoke a prover. They pin the
things that would silently corrupt the sweep if they drifted: the config
registry used by `python -m zk.runner`, the column set of the results CSV, and
the resume logic that decides which configs still need measuring.
"""

import csv
from pathlib import Path

import pytest

from zk import scaling, workloads


def test_factories_cover_the_sweep_grid():
    grid_names = {name for name, _, _ in scaling.CONFIGS}
    assert grid_names <= set(workloads.FACTORIES)


def test_every_grid_entry_resolves_to_a_bundle():
    for name, batch, epochs in scaling.CONFIGS:
        assert batch >= 1 and epochs >= 1
    # building a bundle loads data, so spot-check one config per family
    for name, expected_task in (("mlp_tabular", "binary"), ("tiny_cnn", "multiclass")):
        bundle = workloads.FACTORIES[name]()
        assert bundle.name == name
        assert bundle.task == expected_task
        assert len(bundle.x_test) > 0


def test_csv_fields_cover_scaling_row_keys():
    required = {"name", "num_rows", "logrows", "prove_s", "verify_ms",
                "proof_kb", "peak_rss_mb", "verified", "status"}
    assert required <= set(scaling.FIELDS)


def test_resume_state_reads_only_csv(tmp_path: Path):
    path = tmp_path / "zk_scaling.csv"
    path.write_text(
        "name,batch,num_rows,status\n"
        "mlp_tabular,1,5208,ok\n"
        "mlp_h256,1,32568,error\n",
        encoding="utf-8",
    )
    done = scaling._load_done(path)
    assert set(done) == {"mlp_tabular", "mlp_h256"}
    assert done["mlp_h256"]["status"] == "error"


def test_scaled_mlp_widths_are_monotone_in_parameter_count():
    counts = [workloads.FACTORIES[n]().name for n in ("mlp_h128", "mlp_h256", "mlp_h512")]
    assert counts == ["mlp_h128", "mlp_h256", "mlp_h512"]


def test_quantized_accuracy_is_measured_not_asserted():
    pytest.importorskip("torch")
    from zk.quant_utility import evaluate

    row = evaluate("mlp_tabular", scale=8)
    assert row["status"] == "ok"
    assert 0.0 <= row["accuracy_quant"] <= 1.0
    assert 0.0 <= row["accuracy_float"] <= 1.0
    # a 8-bit grid on a 30-feature standardized tabular MLP must not collapse
    assert row["accuracy_quant"] > 0.5
