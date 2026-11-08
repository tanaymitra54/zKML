"""In-circuit cost of the DP mechanism --- analytical estimates only.

The measured paper circuit keeps the sampler and the accountant *off* circuit.
This module prices the obvious next experiment: move three DP-specific
computations into the relation and estimate the constraint rows they would add.

Everything here is **ESTIMATED** from an explicit unit-cost model.  No row in
this file is a benchmark, and the CSV labels every data row ``ESTIMATED`` so the
distinction cannot be lost downstream.

Units are Halo2/PLONK constraint rows on BN254:

* one Poseidon permutation                     : 270 rows
* fixed-point ``ln`` / ``sqrt`` / ``sin+cos``  : 1200 / 400 / 1600 rows
* one plookup table read                      : 3 rows
* one fixed-point multiply / comparison       : 12 / 18 rows

Output: ``results/incircuit_cost.csv``.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS / "incircuit_cost.csv"

LABEL = "ESTIMATED"

# --- unit-cost assumptions (ESTIMATED, not measured) -----------------------
POSEIDON_ROWS = 270
LN_ROWS = 1200
SQRT_ROWS = 400
SINCOS_ROWS = 1600
LOOKUP_ROWS = 3
RANGE_ROWS = 4
MUL_ROWS = 12
CMP_ROWS = 18

# --- reference workload and parameters -------------------------------------
D = 8
W = 128
M = 64
RDP_ORDERS = 63
PLOOKUP_TABLE_BITS = 12
PLOOKUP_TABLE_SIZE = 1 << PLOOKUP_TABLE_BITS
POSEIDON_RATE = 2
DIGESTS = 7
LIMBS_PER_DIGEST = 8

FIELDS = [
    "component",
    "variant",
    "unit",
    "quantity",
    "rows_per_unit",
    "estimated_rows",
    "label",
    "model",
    "assumptions",
]


def _row(component, variant, unit, quantity, rows_per_unit, model, assumptions):
    return {
        "component": component,
        "variant": variant,
        "unit": unit,
        "quantity": str(quantity),
        "rows_per_unit": str(rows_per_unit),
        "estimated_rows": str(int(quantity * rows_per_unit)),
        "label": LABEL,
        "model": model,
        "assumptions": assumptions,
    }


def box_muller_per_sample() -> int:
    """Two uniforms, one log, one sqrt, one sin/cos per normal sample."""
    uniforms = 2 * POSEIDON_ROWS
    return uniforms + LN_ROWS + SQRT_ROWS + SINCOS_ROWS


def plookup_per_sample() -> int:
    """Two inverse-CDF table reads plus range checks per sample."""
    return 2 * LOOKUP_ROWS + 2 * RANGE_ROWS


def rdp_step_rows() -> int:
    """Per-order arithmetic plus a min-reduction over the RDP order set."""
    per_order = 4 * MUL_ROWS + RANGE_ROWS
    reductions = (RDP_ORDERS - 1) * CMP_ROWS
    return RDP_ORDERS * per_order + reductions


def poseidon_recompute_rows() -> tuple[int, int]:
    """Return (permutations, rows incl. limb packing) for the 7 digests."""
    field_elements = M * D + D * W + DIGESTS  # batch + model + metadata
    permutations = math.ceil(field_elements / POSEIDON_RATE)
    packing = DIGESTS * LIMBS_PER_DIGEST * 2
    return permutations, permutations * POSEIDON_ROWS + packing


def estimate_rows() -> list[dict[str, str]]:
    bm = box_muller_per_sample()
    pl = plookup_per_sample()
    rdp = rdp_step_rows()
    permutations, poseidon_rows = poseidon_recompute_rows()
    table_perms = math.ceil(PLOOKUP_TABLE_SIZE / POSEIDON_RATE)

    rows = [
        _row(
            "discrete-Gaussian sampler",
            "Box-Muller (per sample)",
            "sample",
            M,
            bm,
            f"2*{POSEIDON_ROWS} (uniforms) + {LN_ROWS} (ln) + {SQRT_ROWS} (sqrt) + {SINCOS_ROWS} (sin/cos)",
            "Fixed-point transcendentals approximated by range reduction plus polynomial; "
            "two Poseidon-derived uniforms per sample.",
        ),
        _row(
            "discrete-Gaussian sampler",
            "Box-Muller (per release)",
            "release",
            1,
            M * bm,
            f"{M} samples x {bm} rows/sample",
            "One release draws m=64 independent normals.",
        ),
        _row(
            "discrete-Gaussian sampler",
            "plookup inversion (per sample)",
            "sample",
            M,
            pl,
            f"2 x {LOOKUP_ROWS} (table reads) + 2 x {RANGE_ROWS} (range checks)",
            "Inverse-CDF lookup over a 2^12-entry table; no transcendental circuit.",
        ),
        _row(
            "discrete-Gaussian sampler",
            "plookup inversion (per release)",
            "release",
            1,
            M * pl,
            f"{M} samples x {pl} rows/sample",
            "Assumes the 2^12 table is fixed public parameters, so it is not re-proved.",
        ),
        _row(
            "discrete-Gaussian sampler",
            "plookup table commitment (one-time)",
            "table entry",
            PLOOKUP_TABLE_SIZE,
            POSEIDON_ROWS // POSEIDON_RATE,
            f"ceil({PLOOKUP_TABLE_SIZE}/{POSEIDON_RATE}) = {table_perms} Poseidon permutations",
            "Only charged if the table is committed per proof; fixed public parameters cost 0 rows.",
        ),
        _row(
            "RDP accountant step",
            "per-order arithmetic",
            "order",
            RDP_ORDERS,
            4 * MUL_ROWS + RANGE_ROWS,
            f"per order: 4 x {MUL_ROWS} (mul/div) + {RANGE_ROWS} (range check)",
            "log(1/delta) is a public constant; sigma, delta_f and the per-order sums are private.",
        ),
        _row(
            "RDP accountant step",
            "min-reduction over orders",
            "comparison",
            RDP_ORDERS - 1,
            CMP_ROWS,
            f"{RDP_ORDERS - 1} pairwise min comparisons",
            "Tree reduction over the 63 per-order certificates.",
        ),
        _row(
            "Poseidon commitment recomputation",
            "7 digests over D and theta",
            "permutation",
            permutations,
            POSEIDON_ROWS,
            f"ceil((m*d + d*w + {DIGESTS})/{POSEIDON_RATE}) = {permutations} permutations",
            "Absorbs the full private batch D (m*d) and model theta (d*w); rate 2.",
        ),
        _row(
            "Poseidon commitment recomputation",
            "limb packing for public instance",
            "limb",
            DIGESTS * LIMBS_PER_DIGEST,
            2,
            f"{DIGESTS} digests x {LIMBS_PER_DIGEST} limbs x 2 rows",
            "128-bit digests split into eight 16-bit limbs for the public output.",
        ),
    ]

    bm_total = M * bm + rdp + poseidon_rows
    pl_total = M * pl + rdp + poseidon_rows
    rows.append(
        _row(
            "TOTAL",
            "Box-Muller sampler variant",
            "release",
            1,
            bm_total,
            f"{M * bm} (sampler) + {rdp} (RDP) + {poseidon_rows} (Poseidon)",
            "Sum of the Box-Muller sampler, accountant and commitment recomputation rows.",
        )
    )
    rows.append(
        _row(
            "TOTAL",
            "plookup sampler variant",
            "release",
            1,
            pl_total,
            f"{M * pl} (sampler) + {rdp} (RDP) + {poseidon_rows} (Poseidon)",
            "Assumes the lookup table is fixed public parameters, not re-proved.",
        )
    )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path = CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> Path:
    rows = estimate_rows()
    path = write_csv(rows)
    for row in rows:
        if row["component"] == "TOTAL":
            print(f"{row['variant']}: {row['estimated_rows']} rows ({row['label']})")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
