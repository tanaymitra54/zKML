"""DP utility: held-out accuracy under Gaussian noise at eps in {0.5,1,2,4,8}.

Noise is added to logits before the decision rule (Gaussian mechanism on the
release vector). Sensitivity is the measured max L2 change of the logit vector
under a one-record leave-one-out swap on a probe subset (empirical bound).
"""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from zk.workloads import mlp_tabular, tiny_cnn, train_bundle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "zk_utility.csv"
ELSEVIER = Path(
    "/Users/tanaymitra/Documents/research_paper/karthikeyan/paper/paper/elsevier/results/zk_utility.csv"
)

EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]
DELTA = 1e-5
FIELDS = ["workload", "epsilon", "delta", "sigma", "sensitivity_l2", "accuracy",
          "accuracy_std", "n_trials", "clean_accuracy", "n_test", "seed"]


def _gaussian_sigma(eps: float, delta: float, sensitivity: float) -> float:
    # classic approx: sigma >= sensitivity * sqrt(2 ln(1.25/delta)) / eps
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def _release_vector(model: nn.Module, x: torch.Tensor, task: str) -> np.ndarray:
    """Probability release: sigmoid or softmax (bounded, fixed sensitivity)."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if task == "binary":
            return torch.sigmoid(logits).cpu().numpy()
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _sensitivity(task: str) -> float:
    # Binary probability in [0,1] => L2 sens 1. Softmax simplex => L2 sens sqrt(2).
    return 1.0 if task == "binary" else math.sqrt(2.0)


def _accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    task: str,
    noise_std: float,
    rng: np.random.Generator,
) -> float:
    probs = _release_vector(model, x, task)
    if noise_std > 0:
        probs = probs + rng.normal(0.0, noise_std, size=probs.shape)
    if task == "binary":
        pred = (probs.reshape(-1) >= 0.5).astype(np.float32)
        yt = y.detach().cpu().numpy().reshape(-1)
        return float((pred == yt).mean())
    pred = probs.argmax(axis=1)
    yt = y.detach().cpu().numpy().reshape(-1)
    return float((pred == yt).mean())


def evaluate(seed: int = 0, n_trials: int = 25) -> list[dict]:
    rows: list[dict] = []
    bundles = [
        mlp_tabular(seed=seed),
        tiny_cnn(seed=seed, n_train=8000, n_test=1000),
    ]
    rng = np.random.default_rng(seed)

    for bundle in bundles:
        epochs = 50 if bundle.name == "mlp_tabular" else 40
        lr = 1e-2 if bundle.name == "mlp_tabular" else 5e-4
        model = train_bundle(bundle, epochs=epochs, lr=lr, seed=seed)
        sens = _sensitivity(bundle.task)
        clean = _accuracy(model, bundle.x_test, bundle.y_test, bundle.task, 0.0, rng)
        print(f"{bundle.name} clean_acc={clean:.4f}", flush=True)
        for eps in EPSILONS:
            sigma = _gaussian_sigma(eps, DELTA, sens)
            # One noise draw per epsilon would be noise, not a measurement: a
            # single bad draw can invert the eps ordering. Average n_trials.
            draws = [
                _accuracy(model, bundle.x_test, bundle.y_test, bundle.task, sigma, rng)
                for _ in range(n_trials)
            ]
            acc = float(np.mean(draws))
            rows.append(
                {
                    "workload": bundle.name,
                    "epsilon": eps,
                    "delta": DELTA,
                    "sigma": round(sigma, 6),
                    "sensitivity_l2": round(sens, 6),
                    "accuracy": round(acc, 6),
                    "accuracy_std": round(float(np.std(draws)), 6),
                    "n_trials": n_trials,
                    "clean_accuracy": round(clean, 6),
                    "n_test": int(len(bundle.x_test)),
                    "seed": seed,
                }
            )
            print(f"{bundle.name} eps={eps}: acc={acc:.4f}+-{np.std(draws):.4f} "
                  f"sigma={sigma:.4f} sens={sens:.4f}", flush=True)
    return rows


def write_csv(rows: list[dict], path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ELSEVIER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, ELSEVIER)
    return path


def main() -> Path:
    rows = evaluate()
    path = write_csv(rows)
    print(f"wrote {path}")
    print(f"wrote {ELSEVIER}")
    return path


if __name__ == "__main__":
    main()
