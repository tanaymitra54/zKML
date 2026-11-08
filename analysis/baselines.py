"""Analytical baselines for the DP-zkML release boundary (paper Sec. VII-B).

Three designs are compared, all at the paper's reference workload
``d=8, w=128, m=64`` with noise multiplier ``sigma=4`` and ``T=10`` releases:

* **B1 -- pure zkML**: a proof of the inference aggregate, no differential
  privacy.  Circuit rows are the ML term only (Eq. 12).
* **B2 -- pure DP**: the Gaussian mechanism and accountant run in trusted code;
  there is no proof at all, so circuit rows are zero.
* **B3 -- naive verifiable-noise**: the same DP release, but the sampler, the
  accountant step and the binding layer are all placed inside the circuit, so
  the full Eq. 12 row model applies.

Every row is *analytical*, not a measurement.  Drynx is included only as a
citation: it is never run, never timed, and no number is attributed to it.

Output: ``results/baselines.csv``.
"""
from __future__ import annotations

import csv
from pathlib import Path

from analysis.accounting import rdp_epsilon
from analysis.constraints import constraint_proxy

RESULTS = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS / "baselines.csv"

DEFAULT_D = 8
DEFAULT_W = 128
DEFAULT_M = 64
DEFAULT_B = 8
DEFAULT_BU = 32
DEFAULT_SIGMA = 4.0
DEFAULT_T = 10
DEFAULT_DELTA = 1e-5

FIELDS = [
    "baseline",
    "description",
    "workload",
    "sigma",
    "T",
    "delta",
    "epsilon",
    "ml_rows",
    "dp_rows",
    "circuit_rows",
    "proof_required",
    "dp_noise",
    "row_model",
    "status",
    "source",
    "notes",
]


def build_rows(
    d: int = DEFAULT_D,
    w: int = DEFAULT_W,
    m: int = DEFAULT_M,
    b: int = DEFAULT_B,
    bu: int = DEFAULT_BU,
    sigma: float = DEFAULT_SIGMA,
    T: int = DEFAULT_T,
    delta: float = DEFAULT_DELTA,
) -> list[dict[str, str]]:
    """Return the B1/B2/B3 rows plus the citation-only Drynx row."""
    model = constraint_proxy(d=d, w=w, m=m, b=b, bu=bu, sigma=sigma)
    eps = rdp_epsilon(sigma, T, delta=delta)
    workload = f"d={d},w={w},m={m}"
    ml_rows = model["RML"]
    dp_rows = model["R"] - model["RML"]

    rows = [
        {
            "baseline": "B1",
            "description": "pure zkML: prove inference aggregate, no DP",
            "workload": workload,
            "sigma": str(sigma),
            "T": str(T),
            "delta": str(delta),
            "epsilon": "inf",
            "ml_rows": str(ml_rows),
            "dp_rows": "0",
            "circuit_rows": str(ml_rows),
            "proof_required": "yes",
            "dp_noise": "no",
            "row_model": "Eq. 12 ML-only (R_ML=d*w^2)",
            "status": "analytical",
            "source": "this work, Eq. 12",
            "notes": "No privacy guarantee; epsilon is unbounded.",
        },
        {
            "baseline": "B2",
            "description": "pure DP: trusted Gaussian mechanism and accountant, no proof",
            "workload": workload,
            "sigma": str(sigma),
            "T": str(T),
            "delta": str(delta),
            "epsilon": f"{eps:.4f}",
            "ml_rows": "0",
            "dp_rows": "0",
            "circuit_rows": "0",
            "proof_required": "no",
            "dp_noise": "yes",
            "row_model": "none (no circuit)",
            "status": "analytical",
            "source": "this work, Eq. 4 (RDP accountant)",
            "notes": (
                "RDP bound over T unsampled Gaussian releases at delta=1e-5. "
                "Correctness rests on a trusted executor."
            ),
        },
        {
            "baseline": "B3",
            "description": "naive verifiable-noise: sampler, accountant and binding all in-circuit",
            "workload": workload,
            "sigma": str(sigma),
            "T": str(T),
            "delta": str(delta),
            "epsilon": f"{eps:.4f}",
            "ml_rows": str(ml_rows),
            "dp_rows": str(dp_rows),
            "circuit_rows": str(model["R"]),
            "proof_required": "yes",
            "dp_noise": "yes",
            "row_model": "Eq. 12 full (R_ML+R_range+R_sample+R_account+R_bind)",
            "status": "analytical",
            "source": "this work, Eq. 12",
            "notes": (
                f"R_sample={model['Rsample']}, R_account={model['Raccount']}, "
                f"R_range={model['Rrange']}, R_bind={model['Rbind']}. "
                "Rows scale with sigma through K=ceil(6*sigma); the paper's "
                "sigma=2 reference point gives R=160316."
            ),
        },
        {
            "baseline": "Drynx",
            "description": "decentralised verifiable DP for statistical queries (citation only)",
            "workload": "",
            "sigma": "",
            "T": "",
            "delta": "",
            "epsilon": "",
            "ml_rows": "",
            "dp_rows": "",
            "circuit_rows": "",
            "proof_required": "",
            "dp_noise": "",
            "row_model": "",
            "status": "cited-only; not run",
            "source": "Froelicher et al., IEEE TIFS 15:3035-3050, 2020",
            "notes": (
                "Included as related work only. No timing, row count or proof "
                "size is claimed for Drynx in this repository."
            ),
        },
    ]
    return rows


def write_csv(rows: list[dict[str, str]], path: Path = CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> Path:
    rows = build_rows()
    path = write_csv(rows)
    for row in rows:
        if row["baseline"] == "Drynx":
            continue
        print(
            f"{row['baseline']}: rows={row['circuit_rows']:>7} "
            f"epsilon={row['epsilon']:>8} proof={row['proof_required']}"
        )
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
