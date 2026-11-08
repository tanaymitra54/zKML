"""Quantization utility: float vs fixed-point accuracy -> ``results/zk_quant_utility.csv``.

The circuit cannot carry float32 weights. EZKL maps everything onto a fixed-point
grid of ``input_scale`` / ``param_scale`` bits inside a 16-bit lookup domain. This
module measures what that snapping costs in accuracy, which is the utility side
of the cost/utility trade-off the paper reports.

Method (per workload):

1. train the same model the sweep exports (same seed and epoch budget),
2. evaluate float32 accuracy on the held-out split,
3. snap weights **and** inputs to ``round(v * 2**scale) / 2**scale`` clipped to
   the same int16 domain EZKL uses, evaluate again,
4. report both accuracies, their difference, and the prediction agreement rate.

This is a *simulation of the quantization grid*, not a proving run. The real
circuit's output drift is separately recorded per config in
``results/zk_scaling.csv`` (``max_abs_output_drift``) where it is available.

    python -m zk.quant_utility                 # default workloads, scale 8
    python -m zk.quant_utility --scale 12 --configs mlp_tabular
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from zk import workloads as W

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "zk_quant_utility.csv"
ELSEVIER = Path(
    "/Users/tanaymitra/Documents/research_paper/karthikeyan/paper/paper/elsevier/results/zk_quant_utility.csv"
)

FIELDS = [
    "workload", "scale_bits", "epochs", "n_params", "split", "n_test",
    "accuracy_float", "accuracy_quant", "accuracy_delta", "agreement",
    "max_logit_delta", "status", "error",
]

DEFAULTS = ["mlp_tabular", "mlp_h256", "tiny_cnn", "cnn_c8_16_fc64"]
EPOCHS = {"mlp_tabular": 3, "mlp_h128": 2, "mlp_h256": 2, "mlp_h512": 2,
          "tiny_cnn": 2, "cnn_c4_8_fc32": 2, "cnn_c8_16_fc64": 2,
          "cnn_c16_32_fc128": 2}


def _snap(a: np.ndarray, scale: int, lo: float, hi: float) -> np.ndarray:
    step = 2.0 ** -scale
    return np.clip(np.round(a / step) * step, lo, hi).astype(np.float32)


def _forward(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(x).detach().cpu().numpy()


def evaluate(name: str, scale: int = 8, seed: int = 0) -> dict:
    bundle = W.FACTORIES[name]()
    epochs = EPOCHS.get(name, 2)
    model = W.train_bundle(bundle, epochs=epochs, seed=seed)

    x, y = bundle.x_test, bundle.y_test
    logits_f = _forward(model, x)

    lo, hi = -32768 / 2 ** scale, 32767 / 2 ** scale
    q_state = {k: torch.from_numpy(_snap(v.detach().cpu().numpy(), scale, lo, hi))
               for k, v in model.state_dict().items()}
    from copy import deepcopy

    model_q = deepcopy(model)
    model_q.load_state_dict(q_state)
    model_q.eval()
    x_q = torch.from_numpy(_snap(x.detach().cpu().numpy(), scale, lo, hi))
    logits_q = _forward(model_q, x_q)

    def acc(logits: np.ndarray) -> float:
        if bundle.task == "binary":
            pred = (logits.reshape(-1) >= 0).astype(np.int64)
        else:
            pred = logits.argmax(axis=1)
        return float((pred == y.detach().cpu().numpy().reshape(-1)).mean())

    def pred(logits: np.ndarray) -> np.ndarray:
        return (logits.reshape(-1) >= 0).astype(np.int64) if bundle.task == "binary" \
            else logits.argmax(axis=1)

    acc_f, acc_q = acc(logits_f), acc(logits_q)
    return {
        "workload": name,
        "scale_bits": scale,
        "epochs": epochs,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "split": "test",
        "n_test": int(len(y)),
        "accuracy_float": round(acc_f, 4),
        "accuracy_quant": round(acc_q, 4),
        "accuracy_delta": round(acc_q - acc_f, 4),
        "agreement": round(float((pred(logits_f) == pred(logits_q)).mean()), 4),
        "max_logit_delta": round(float(np.max(np.abs(logits_f - logits_q))), 4),
        "status": "ok",
        "error": "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--scale", type=int, default=8)
    a = ap.parse_args(argv)

    names = a.configs or DEFAULTS
    rows = []
    for name in names:
        try:
            row = evaluate(name, scale=a.scale)
            print(f"{name}: float={row['accuracy_float']:.4f} "
                  f"quant={row['accuracy_quant']:.4f} delta={row['accuracy_delta']:+.4f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - recorded honestly
            row = {k: "" for k in FIELDS}
            row.update({"workload": name, "scale_bits": a.scale, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {exc}", flush=True)
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if ELSEVIER.parent.is_dir():
        import shutil

        shutil.copy2(OUT, ELSEVIER)
    print(json.dumps({"path": str(OUT), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
