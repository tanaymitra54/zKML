"""Quantized aggregate reference query (paper Sec. III-B, V-B).

The paper's narrow, fully-specified starting workload: a clipped, quantized
integer aggregate g_theta(D) = sum_r v_theta(r), with ||v_theta(r)||_2 <= CQ
enforced after quantization, giving replacement sensitivity <= 2*CQ.
"""
from __future__ import annotations

import numpy as np


def quantize(x: np.ndarray, scale: float, bits: int = 8) -> np.ndarray:
    """Q_b(a) = clip(round(scale * a)) to signed `bits`-wide integers."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return np.clip(np.round(scale * np.asarray(x, dtype=float)), lo, hi).astype(np.int64)


def clip_vector(v: np.ndarray, CQ: float) -> np.ndarray:
    """Enforce ||v||_2 <= CQ by radial clipping (no-op inside the ball)."""
    n = float(np.linalg.norm(v))
    if CQ <= 0:
        raise ValueError("CQ must be > 0")
    return v if n <= CQ else (v * (CQ / n)).astype(v.dtype)


def clipped_aggregate(records: np.ndarray, v_theta, CQ: float) -> np.ndarray:
    """g_theta(D) = sum_r clip(v_theta(r)) with per-record L2 bound CQ."""
    V = np.stack([clip_vector(np.asarray(v_theta(r), dtype=np.int64), CQ) for r in records])
    return V.sum(axis=0)


def sensitivity_bound(CQ: float) -> float:
    """Replacement sensitivity of the aggregate is at most 2*CQ (Sec. III-B)."""
    return 2.0 * CQ


def field_lower_bound(k: int, A: int, W: int, B: int) -> int:
    """Sufficient field lower bound: p > 2(k*A*W + B) (Sec. V-B)."""
    return 2 * (k * A * W + B)


def synthetic_records(n: int, dim: int, seed: int = 0, spread: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, spread, size=(n, dim)).astype(np.float32)


if __name__ == "__main__":
    CQ = 8.0
    records = synthetic_records(n=256, dim=16, seed=0)
    v_theta = lambda r: np.round(r * 4).astype(np.int64)  # fixed regressor, no training
    z = clipped_aggregate(records, v_theta, CQ)
    assert z.dtype == np.int64 and z.shape == (16,)

    # sensitivity check: any single-record replacement moves the sum by <= 2*CQ
    base = list(records)
    alt = list(records)
    alt[0] = synthetic_records(1, 16, seed=999)[0]
    d = np.linalg.norm(
        clipped_aggregate(np.array(base), v_theta, CQ)
        - clipped_aggregate(np.array(alt), v_theta, CQ)
    )
    assert d <= sensitivity_bound(CQ) + 1e-6, d

    # field bound sanity for a dot product with fan-in k
    p_min = field_lower_bound(k=128, A=127, W=127, B=255)
    assert p_min == 2 * (128 * 127 * 127 + 255)
    print("quant ok: |z|1", int(np.abs(z).sum()), "sens", sensitivity_bound(CQ), "p>", p_min)
