"""Resumable EZKL scaling sweep -> ``results/zk_scaling.csv``.

Each config is measured in a fresh ``python -m zk.runner`` subprocess so
``peak_rss_mb`` is per-run (it does not inherit a larger earlier run's high-water
mark). The CSV is the resume state: on start-up every ``status=ok`` row is
loaded and its config skipped.

The grid is ordered small -> large and is capped at ~5x10^5 circuit rows: past
that point one proving key reaches multiple GB and a single config costs hours
on an M-series laptop, which buys the paper nothing.

    python -m zk.scaling              # fill the grid, stop once >= 50k rows
    python -m zk.scaling --all        # fill the whole grid
    python -m zk.scaling --check      # report done / missing
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "zk_scaling.csv"
ELSEVIER = Path(
    "/Users/tanaymitra/Documents/research_paper/karthikeyan/paper/paper/elsevier/results/zk_scaling.csv"
)
ARTIFACTS = ROOT / "results" / "artifacts" / "scaling"

# Ordered small -> large. (name, batch, epochs)
# The 256-1024 wide MLPs bracket 3x10^4..2x10^5 rows; the CNNs bracket
# 2x10^4..5x10^5. Bigger CNN variants exist in zk.workloads.FACTORIES but are
# deliberately excluded from the default grid (hours per config, GB-class keys).
CONFIGS = [
    ("mlp_tabular", 1, 3),
    ("mlp_h128", 1, 2),
    ("mlp_h256", 1, 2),
    ("mlp_h384", 1, 2),
    ("mlp_h512", 1, 2),
    ("mlp_h768", 1, 1),
    ("mlp_h1024", 1, 1),
    ("tiny_cnn", 1, 2),
    ("cnn_c4_8_fc32", 1, 2),
    ("cnn_c8_16_fc64", 1, 2),
    ("cnn_c16_32_fc128", 1, 2),
]

FIELDS = [
    "name", "batch", "num_rows", "logrows", "n_params", "input_dim",
    "settings_s", "compile_s", "setup_s", "witness_s", "prove_s", "verify_ms",
    "proof_kb", "peak_rss_mb", "verified", "scale_bits", "attempts",
    "work_dir", "wall_s", "system", "status", "error",
]


def _load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["name"]: row for row in csv.DictReader(fh) if row.get("name")}


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _mirror(path: Path) -> None:
    """Copy the CSV into the manuscript's results dir (CSV only, never .tex)."""
    try:
        ELSEVIER.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ELSEVIER)
    except OSError:
        pass


def _measure_subprocess(name: str, work: Path, batch: int, epochs: int, timeout: int) -> dict:
    cmd = [sys.executable, "-m", "zk.runner", "--config", name, "--work-dir", str(work),
           "--batch", str(batch), "--epochs", str(epochs)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": name, "batch": batch, "status": "error",
                "error": f"timeout after {timeout}s", "work_dir": str(work),
                "wall_s": round(time.perf_counter() - t0, 1)}
    wall = round(time.perf_counter() - t0, 1)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"name": name, "batch": batch, "status": "error", "wall_s": wall,
                "error": f"no_json_stdout rc={proc.returncode} stderr={proc.stderr[-500:]}",
                "work_dir": str(work)}
    row = json.loads(lines[-1])
    for k in FIELDS:
        row.setdefault(k, "")
    if row.get("verified") != "":
        row["verified"] = str(row["verified"]).lower()
    row["wall_s"] = wall
    row.setdefault("system", subprocess_result_system())
    if proc.returncode != 0 and row.get("status") == "ok":
        row["status"] = "error"
        row["error"] = (proc.stderr or "").strip().splitlines()[-1][:500]
    return row


def subprocess_result_system() -> str:
    import platform

    return f"{platform.system()} {platform.machine()} / py{platform.python_version()}"


def run(min_rows: int = 50_000, stop_on_target: bool = True,
        only: list[str] | None = None, timeout: int = 3600, force: bool = False) -> Path:
    # Single-writer lock: a second agent/process must not append to the same CSV.
    import fcntl

    lock_path = ROOT / "results" / ".zk_scaling.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another sweep holds {lock_path}; refusing to run concurrently")

    done = {} if force else _load_done(OUT)
    order = [c[0] for c in CONFIGS]
    rows: list[dict] = []
    max_rows = 0
    for name in order:
        row = done.get(name)
        if row and row.get("status") == "ok":
            rows.append(row)
            try:
                max_rows = max(max_rows, int(float(row.get("num_rows") or 0)))
            except ValueError:
                pass

    selected = [c for c in CONFIGS if only is None or c[0] in only]
    unknown = set(only or []) - {c[0] for c in CONFIGS}
    if unknown:
        raise SystemExit(f"unknown configs {sorted(unknown)}; grid is {order}")

    for name, batch, epochs in selected:
        if name in done and done[name].get("status") == "ok" and not force:
            print(f"skip {name} (cached rows={done[name]['num_rows']})", flush=True)
            continue
        if stop_on_target and max_rows >= min_rows:
            print(f"target reached ({max_rows} >= {min_rows}); stopping", flush=True)
            break

        print(f"=== measuring {name} (batch={batch}, epochs={epochs}) ===", flush=True)
        work = ARTIFACTS / name
        row = _measure_subprocess(name, work, batch, epochs, timeout)
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = [r for r in rows if r.get("name") != name]
        rows.append(row)
        rows.sort(key=lambda r: order.index(r["name"]) if r.get("name") in order else 99)
        _write(rows, OUT)
        _mirror(OUT)

        if row.get("status") == "ok":
            max_rows = max(max_rows, int(float(row["num_rows"])))
            print(f"{name}: rows={row['num_rows']} logrows={row['logrows']} "
                  f"prove={row['prove_s']}s verify={row['verify_ms']}ms "
                  f"rss={row['peak_rss_mb']}MB verified={row['verified']}", flush=True)
        else:
            print(f"FAIL {name}: {row.get('error', '')[:300]}", flush=True)

    if max_rows < min_rows:
        print(f"WARNING: max num_rows={max_rows} < target {min_rows}", flush=True)
    else:
        print(f"OK: max num_rows={max_rows} >= {min_rows}", flush=True)
    _mirror(OUT)
    return OUT


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="do not stop at the target")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--target", type=int, default=50_000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--force", action="store_true", help="re-measure even if cached")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    if a.check:
        done = _load_done(OUT)
        ok = sorted(n for n, r in done.items() if r.get("status") == "ok")
        print(json.dumps({
            "path": str(OUT),
            "ok": ok,
            "missing": [n for n, _, _ in CONFIGS if n not in ok],
            "max_rows": max((int(float(done[n].get("num_rows") or 0)) for n in ok), default=0),
        }, indent=2))
        return OUT

    return run(a.target, stop_on_target=not a.all, only=a.configs,
               timeout=a.timeout, force=a.force)


if __name__ == "__main__":
    main()
