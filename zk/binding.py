"""Measured bare/bound binding-layer harness -> ``results/zk_binding.csv``.

Rebuilds the public-instance binding layer the Elsevier paper reports in
Table~tab:baseline, on top of the same EZKL 23.0.5 pipeline as ``zk.runner``.

Recovered circuit shape
-----------------------
The paper's five bare row counts (1,356 / 2,524 / 4,860 / 5,048 / 9,720) are
reproduced exactly by

    release:  y = clip_{[-4096,4096]}( sum_r clip_{[-c,c]}(v_r) + eta )

with ``c = floor(8 / sqrt(d))``, ``x`` private batch data and ``eta`` a separate
d-vector input.  The bound variant appends the 58-value public instance
(7 digests x 8 limbs + noise multiplier + clipping bound) as an identity
passthrough on the proof's public output:

    release:  y = clip( sum_r clip(v_r) + eta ),   out = cat(y, public_58)

That construction reproduces the paper's five bound row counts
(2,255 / 3,423 / 5,759 / 5,947 / 10,619) exactly, and the constant delta of
899 rows.  It is therefore the layer the paper measured: the rows are structural
(the output instance grows by 58 values), not a function of the public values.

Honesty notes
-------------
* ``proof_kb`` is the size of EZKL's ``proof.json`` on disk (the convention used
  by ``zk.runner``).  The paper's 20.36 / 28.60 KB are *not* reproduced by this
  file-size convention on this build; both the declared and the measured values
  are reported in the accompanying message, and no value is adjusted to match.
* Each row is one build and one prove/verify run, in its own subprocess, so
  ``peak_rss_mb`` is per-run (it does not inherit an earlier run's high-water
  mark).  The synthetic points are single runs; the paper's timings are means of
  three.  Rows are deterministic, so a single build is enough for the row counts.
* Public-instance values are fed as normalised fixed-point fractions in [0, 1).
  Row cost depends on the 58-value passthrough, not on the values themselves.

Usage
-----
    python -m zk.binding               # fill results/zk_binding.csv
    python -m zk.binding --check       # report what is already measured
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from zk.runner import measure_prepared

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "zk_binding.csv"
ARTIFACTS = ROOT / "results" / "artifacts" / "binding"
ELSEVIER = Path(
    "/Users/tanaymitra/Documents/research_paper/karthikeyan/paper/paper/elsevier/results/zk_binding.csv"
)

# (name, m, d) for the paper's synthetic sweep.
SYNTHETIC: list[tuple[str, int, int]] = [
    ("syn_8x4", 8, 4),
    ("syn_16x4", 16, 4),
    ("syn_32x4", 32, 4),
    ("syn_16x8", 16, 8),
    ("syn_32x8", 32, 8),
]
# Plus the mid-size realistic circuit the reviewer asked to control.
REALISTIC = ["mlp_h384"]

CONFIGS = [n for n, _, _ in SYNTHETIC] + REALISTIC
MODES = ("bare", "bound")

FIELDS = [
    "config", "rows_bare", "rows_bound", "delta_rows",
    "proof_kb_bare", "proof_kb_bound", "prove_s", "verify_ms", "peak_rss_mb",
]


class SyntheticBare(nn.Module):
    """z = sum_r clip(v_r); y = clip(z + eta)."""

    def __init__(self, m: int, d: int, c_q: float = 8.0, out_bound: float = 4096.0):
        super().__init__()
        self.m, self.d = m, d
        self.c = math.floor(c_q / math.sqrt(d))
        self.out_bound = out_bound

    def forward(self, x: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        z = x.reshape(self.m, self.d).clamp(-self.c, self.c).sum(0)
        return (z + eta).clamp(-self.out_bound, self.out_bound)


class SyntheticBound(nn.Module):
    """Bare release plus the 58-value public-instance passthrough."""

    def __init__(self, m: int, d: int, n_public: int = 58):
        super().__init__()
        self.bare = SyntheticBare(m, d)
        self.n_public = n_public

    def forward(self, x: torch.Tensor, eta: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.bare(x, eta), public], dim=0)


class BoundWrapper(nn.Module):
    """Append the public instance to an arbitrary model's release vector."""

    def __init__(self, base: nn.Module, n_public: int = 58):
        super().__init__()
        self.base = base
        self.n_public = n_public

    def forward(self, x: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.base(x).reshape(-1), public], dim=0)


def public_instance(n: int = 58, seed: int = 0) -> np.ndarray:
    """Normalised 16-bit-ish limb values for 7 digests + sigma + clip bound."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=n, dtype=np.int64).astype(np.float64) / 65535.0


# --------------------------------------------------------------------------
# one measurement
# --------------------------------------------------------------------------
def _measure_bare_or_bound(name: str, mode: str, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    if name in dict((n, (m, d)) for n, m, d in SYNTHETIC):
        m, d = dict((n, (m, d)) for n, m, d in SYNTHETIC)[name]
        rng = np.random.default_rng(0)
        c = math.floor(8 / math.sqrt(d))
        x = torch.from_numpy(rng.integers(-c, c + 1, size=m * d).astype(np.float32))
        eta = torch.from_numpy(rng.integers(-2, 3, size=d).astype(np.float32))
        pub = torch.from_numpy(public_instance(58).astype(np.float32))
        if mode == "bare":
            model, inputs, names = SyntheticBare(m, d), [x, eta], ["input", "eta"]
        else:
            model, inputs, names = SyntheticBound(m, d), [x, eta, pub], ["input", "eta", "public"]
    else:
        # realistic model: same weights for bare and bound, trained once per run.
        from zk.workloads import mlp_scaled, sample_input, train_bundle

        bundle = mlp_scaled(int(name.removeprefix("mlp_h")))
        base = train_bundle(bundle, epochs=2, seed=0)
        x = sample_input(bundle, batch=1)
        pub = torch.from_numpy(public_instance(58).astype(np.float32))
        if mode == "bare":
            model, inputs, names = base, [x], ["input"]
        else:
            model, inputs, names = BoundWrapper(copy.deepcopy(base)), [x, pub], ["input", "public"]

    res = measure_prepared(
        model, inputs, work_dir,
        input_names=names, output_names=["output"],
        train_epochs=0, calibrate=False, drop_proving_key=True,
    )
    row = res if isinstance(res, dict) else res.__dict__
    row["mode"] = mode
    row["config"] = name
    return row


def _cli_one(name: str, mode: str, work_dir: Path) -> int:
    row = _measure_bare_or_bound(name, mode, work_dir)
    print(json.dumps(row))
    return 0 if row.get("verified") else 1


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def _run_subprocess(name: str, mode: str, work: Path, timeout: int) -> dict:
    cmd = [sys.executable, "-m", "zk.binding", "--one", name, "--mode", mode,
           "--work-dir", str(work)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"config": name, "mode": mode, "status": "error",
                "error": f"timeout after {timeout}s", "wall_s": round(time.perf_counter() - t0, 1)}
    wall = round(time.perf_counter() - t0, 1)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"config": name, "mode": mode, "status": "error", "wall_s": wall,
                "error": f"no_json rc={proc.returncode} stderr={proc.stderr[-500:]}"}
    row = json.loads(lines[-1])
    row["wall_s"] = wall
    row.setdefault("system", f"{platform.system()} {platform.machine()} / py{platform.python_version()}")
    if proc.returncode != 0:
        row["status"] = "error"
        row.setdefault("error", (proc.stderr or "").strip().splitlines()[-1][:500] if proc.stderr else "")
    return row


def _load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["config"]: r for r in csv.DictReader(fh) if r.get("config")}


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _mirror(path: Path) -> None:
    try:
        ELSEVIER.parent.mkdir(parents=True, exist_ok=True)
        ELSEVIER.write_bytes(path.read_bytes())
    except OSError:
        pass


def run(timeout: int = 1800, force: bool = False, only: list[str] | None = None) -> Path:
    done = {} if force else _load_done(OUT)
    selected = [n for n in CONFIGS if only is None or n in only]
    rows: list[dict] = []
    for name in selected:
        if name in done and done[name].get("rows_bound") and not force:
            rows.append(done[name])
            print(f"skip {name} (cached)", flush=True)
            continue
        pair: dict[str, dict] = {}
        for mode in MODES:
            print(f"=== {name} [{mode}] ===", flush=True)
            pair[mode] = _run_subprocess(name, mode, ARTIFACTS / name / mode, timeout)
            print(json.dumps({k: pair[mode].get(k) for k in
                              ("rows", "num_rows", "logrows", "prove_s", "verify_ms",
                               "proof_kb", "peak_rss_mb", "status")}), flush=True)
        b, d = pair["bare"], pair["bound"]
        if b.get("status") == "error" or d.get("status") == "error":
            row = {"config": name, "rows_bare": b.get("num_rows", ""),
                   "rows_bound": d.get("num_rows", ""), "delta_rows": "",
                   "proof_kb_bare": b.get("proof_kb", ""), "proof_kb_bound": d.get("proof_kb", ""),
                   "prove_s": d.get("prove_s", ""), "verify_ms": d.get("verify_ms", ""),
                   "peak_rss_mb": d.get("peak_rss_mb", "")}
        else:
            rb, rd = int(b["num_rows"]), int(d["num_rows"])
            row = {
                "config": name,
                "rows_bare": rb, "rows_bound": rd, "delta_rows": rd - rb,
                "proof_kb_bare": b["proof_kb"], "proof_kb_bound": d["proof_kb"],
                "prove_s": d["prove_s"], "verify_ms": d["verify_ms"],
                "peak_rss_mb": d["peak_rss_mb"],
            }
        rows = [r for r in rows if r.get("config") != name]
        rows.append(row)
        rows.sort(key=lambda r: CONFIGS.index(r["config"]) if r.get("config") in CONFIGS else 99)
        _write(rows, OUT)
        _mirror(OUT)
    _mirror(OUT)
    return OUT


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one", metavar="CONFIG", default=None)
    ap.add_argument("--mode", choices=MODES, default="bound")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    if a.one:
        wd = a.work_dir or (ARTIFACTS / a.one / a.mode)
        return _cli_one(a.one, a.mode, wd)  # type: ignore[return-value]

    if a.check:
        done = _load_done(OUT)
        print(json.dumps({"path": str(OUT), "done": sorted(done),
                          "missing": [c for c in CONFIGS if c not in done]}, indent=2))
        return OUT

    unknown = set(a.configs or []) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs {sorted(unknown)}; known {CONFIGS}")
    return run(timeout=a.timeout, force=a.force, only=a.configs)


if __name__ == "__main__":
    main()
