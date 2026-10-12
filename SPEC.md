# DP–zkML reference implementation — SPEC

Source paper: `zMKL1.2 (1).pdf` — *Zero-Knowledge Machine Learning with
Differential Privacy* (Mitra, Adithyan, Sagnik, Karthikeyan). This repo turns the
paper's analytical study + Section IX blueprint into runnable code.

Status of the paper: **design study, no implemented system.** So we build, in
order: (A) reproduce the analytical evaluation, (B) DP core, (C) quantized
reference query, (D) the ZK relation + verifier.

## Track ownership (avoid edit collisions)

| Track | Dir | Owner | Deliverable |
|-------|-----|-------|-------------|
| A. Analytical reproduction | `analysis/` | cursor | Figs 3–9 + acceptance tests |
| B. DP core (sampler/accountant/ledger) | `dp/` | opencode | discrete Gaussian, RDP/zCDP, release ledger |
| C. Quantized reference query | `quant/` | opencode | int8 aggregate, clipping, sensitivity |
| D. ZK relation + verifier | `zk/` | joint | prove & verify Algo 1 |

Tests: put track tests under `tests/`, prefix with the track letter
(`tests/test_a_*.py`, `tests/test_b_*.py`). Do not edit another track's dir.

## Global acceptance numbers (from the paper — must match)

Accounting, Eq. (4), unsampled Gaussian, Δ=1, δ=1e-5, orders A={2..64},
`r_t(α) = αΔ²/(2σ²)`:
- T=10, σ=4  → ε ≈ **4.106**
- T=40, σ=4  → ε ≈ **8.838**
- max releases at ε ≤ 8, σ=4 → **33**

Constraint model, Eq. (12): `RML=dw²`, `Rrange=2dwb`,
`Rsample=m(2bu+ceil(log2(2K+1)))`, `Raccount=4|A|`, `Rbind=8192`.
- default d=8, w=128, m=64, b=8, bu=32, σ=2 (K=ceil(6σ)=12):
  RML=131072, Rrange=16384, Rsample=4416, Raccount=252, Rbind=8192,
  **Total=160316**, overhead vs ML-only ≈ **22.3%**.

Proving-work index, Eq. (13): `IP(R) = R log2 R / (R0 log2 R0)`,
`R0 = 8·128² = 131072`. Default → IP ≈ **1.244**.

Utility proxy, Eq. (14): `U(c,σ)=clip(1 − 0.10/√c − 0.12·log(1+σ²), 0, 1)`.

Verification model, Eq. (15): `log(IV) ~ N(log(1+0.003·s), 0.18²)`.

zCDP reference, Eq. (5): `ϱ_t=Δ²/(2σ²)`, `ε_zCDP = ϱ_tot + √(2·ϱ_tot·log(1/δ))`.

## Interfaces (frozen — both tracks code to these)

```python
# analysis/accounting.py
def rdp_epsilon(sigma: float, T: int, delta: float = 1e-5,
                orders: list[int] = list(range(2, 65)), delta_f: float = 1.0) -> float: ...
def max_releases(sigma: float, eps_cap: float, delta: float = 1e-5) -> int: ...
def zcdp_epsilon(sigma: float, T: int, delta: float = 1e-5, delta_f: float = 1.0) -> float: ...

# analysis/constraints.py
def constraint_proxy(d:int,w:int,m:int=64,b:int=8,bu:int=32,sigma:float=2.0,
                     orders=list(range(2,65)), bind:int=8192) -> dict[str,int]: ...
    # returns {"RML","Rrange","Rsample","Raccount","Rbind","R","overhead_pct"}
def proving_work_index(R:int, R0:int=131072) -> float: ...

# analysis/utility.py
def utility(c: float, sigma: float) -> float: ...
def calibrated_utility(c: float, eps: float, T: int = 1, delta: float = 1e-5) -> float: ...
    # bisection: find sigma s.t. rdp_epsilon(sigma,T)==eps, then utility(c,sigma)

# dp/sampler.py
def discrete_gaussian(sigma: float, size: int, rng) -> "np.ndarray": ...
    # exact (Canonne-Kamath-Steinke) via integer thresholds; expose pmf support
def sampler_pmf(bit_width: int, sigma: float) -> "dict[int,float]": ...  # exhaustive for small bits

# dp/accountant.py
class Accountant:  # interval-valued, conservative
    def charge(self, sigma: float, delta_f: float = 1.0) -> None: ...
    def epsilon(self, delta: float = 1e-5) -> tuple[float,float]: ...  # (lo, hi)

# dp/ledger.py  (SQLite, append-only, single-successor)
class ReleaseLedger:
    def __init__(self, path: str): ...
    def propose(self, release_id: str, prev_root: str, account_state: dict) -> str: ...
    def commit(self, release_id: str) -> str: ...   # exactly one successor per state
    def root(self) -> str: ...

# quant/reference_query.py
def quantize(x, scale: float, bits: int = 8) -> "np.ndarray": ...
def clipped_aggregate(records, v_theta, CQ: float) -> "np.ndarray": ...
def sensitivity_bound(CQ: float) -> float:  # 2*CQ
def field_lower_bound(k:int,A:int,W:int,B:int) -> int:  # p > 2(kAW+B)
```

## Metrics naming rule (paper Sec. VII-A)
Never label a synthetic ordinate as measured accuracy / ms / proof size.
Figures from Track A are "analytical", not benchmarks.

## Dataset
All analytical + circuit work uses **synthetic** data (fixed-size record set,
bounded integer embeddings ‖vθ(r)‖∞ ≤ CQ), per Sec. III. Optional real workload
(MNIST int8 MLP) is a later, clearly-labeled variant — do not add it to Track A.
