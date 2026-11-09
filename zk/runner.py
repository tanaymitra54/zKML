"""One measured EZKL configuration: compile -> setup -> prove -> verify.

Every number produced here is a real local measurement on this machine.
Nothing is estimated or interpolated.

Design notes
------------
* EZKL 23.0.5 API: ``setup`` and ``gen_witness`` take the *compiled circuit*
  path, and ``ezkl.get_srs`` is async (it drives a Rust tokio runtime), so it
  must be awaited inside a running loop.
* The KZG SRS is cached once per ``logrows`` under ``zk/data/srs`` and shared by
  every configuration, so the sweep does not re-download it per config.
* ``peak_rss_mb`` is the max RSS of this process. ``zk.scaling`` runs one config
  per subprocess, so the value belongs to that config alone.
* **Fixed-point overflow retry.** Wide MLPs occasionally calibrate to scales
  where the witnessed circuit exceeds the range-check domain and halo2 aborts
  with "The constraint system is not satisfied". That is a quantization
  property, not a harness bug. When it happens we re-run the *whole* pipeline
  with explicit wider scales (12, then 14 bits) and record which scale finally
  produced a verifiable proof in ``scale_bits`` / ``attempts``. A config whose
  every attempt fails is reported with its real error — never filled in.
"""
from __future__ import annotations

import asyncio
import json
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import ezkl
import torch

from zk.workloads import WorkloadBundle, export_onnx, sample_input, train_bundle

SRS_DIR = Path(__file__).resolve().parent / "data" / "srs"

# None = whatever gen_settings/calibrate_settings chooses; then explicit widths.
SCALE_ATTEMPTS: tuple[int | None, ...] = (None, 12, 14)


@dataclass
class RunResult:
    name: str
    batch: int
    num_rows: int
    logrows: int
    n_params: int
    input_dim: int
    settings_s: float
    compile_s: float
    setup_s: float
    witness_s: float
    prove_s: float
    verify_ms: float
    proof_kb: float
    peak_rss_mb: float
    verified: bool
    work_dir: str
    scale_bits: int = 0
    attempts: int = 1
    status: str = "ok"
    error: str = ""


def _rss_mb() -> float:
    """Max RSS of this process. macOS reports bytes, Linux kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024.0


def _srs(logrows: int) -> Path:
    """Fetch-and-cache the KZG structured reference string for this logrows."""
    SRS_DIR.mkdir(parents=True, exist_ok=True)
    path = SRS_DIR / f"kzg_{logrows}.srs"
    if path.is_file() and path.stat().st_size > 0:
        return path

    async def _fetch() -> None:
        await ezkl.get_srs(srs_path=str(path), logrows=logrows)

    try:
        asyncio.run(_fetch())
    except Exception:
        # Offline fallback: generate a local SRS (slower, still valid).
        ezkl.gen_srs(str(path), logrows)
    return path


def _attempt(
    *,
    onnx_path: Path,
    data_path: Path,
    work_dir: Path,
    logrows_hint: bool,
    scale_bits: int | None,
    calibrate: bool,
    drop_proving_key: bool,
    tag: str,
    run_args=None,
) -> dict:
    """Run the full pipeline once at one scale setting. Raises on failure."""
    suffix = f"_{tag}" if tag else ""
    settings_path = work_dir / f"settings{suffix}.json"
    compiled = work_dir / f"model{suffix}.compiled"
    vk_path = work_dir / f"vk{suffix}.key"
    pk_path = work_dir / f"pk{suffix}.key"
    witness_path = work_dir / f"witness{suffix}.json"
    proof_path = work_dir / f"proof{suffix}.json"

    t0 = time.perf_counter()
    if run_args is not None:
        if not ezkl.gen_settings(str(onnx_path), str(settings_path), py_run_args=run_args):
            raise RuntimeError("gen_settings returned False")
    elif scale_bits is None:
        ezkl.gen_settings(str(onnx_path), str(settings_path))
    else:
        args = ezkl.PyRunArgs()
        args.input_visibility = "public"
        args.param_visibility = "fixed"
        args.input_scale = int(scale_bits)
        args.param_scale = int(scale_bits)
        args.scale_rebase_multiplier = 1
        if not ezkl.gen_settings(str(onnx_path), str(settings_path), py_run_args=args):
            raise RuntimeError("gen_settings returned False")
    settings_s = time.perf_counter() - t0

    if calibrate and scale_bits is None:
        try:
            ezkl.calibrate_settings(str(data_path), str(onnx_path), str(settings_path),
                                    "resources")
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            print(f"warn: calibrate_settings failed ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    t0 = time.perf_counter()
    ezkl.compile_circuit(str(onnx_path), str(compiled), str(settings_path))
    compile_s = time.perf_counter() - t0

    settings = json.loads(settings_path.read_text())
    num_rows = int(settings["num_rows"])
    logrows = int(settings["run_args"]["logrows"])
    srs_path = _srs(logrows)

    t0 = time.perf_counter()
    ezkl.setup(str(compiled), str(vk_path), str(pk_path), srs_path=str(srs_path))
    setup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if not ezkl.gen_witness(str(data_path), str(compiled), str(witness_path)):
        raise RuntimeError("gen_witness returned no witness")
    witness_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ezkl.prove(str(witness_path), str(compiled), str(pk_path), str(proof_path),
               srs_path=str(srs_path))
    prove_s = time.perf_counter() - t0
    peak = _rss_mb()

    t0 = time.perf_counter()
    verified = bool(ezkl.verify(str(proof_path), str(settings_path), str(vk_path),
                                srs_path=str(srs_path)))
    verify_ms = (time.perf_counter() - t0) * 1000.0
    proof_kb = proof_path.stat().st_size / 1024.0

    if drop_proving_key:
        # Multi-GB at these sizes and regenerated by ezkl.setup; verify needs
        # only vk + settings, both tiny.
        try:
            pk_path.unlink()
        except OSError:
            pass

    if not verified:
        raise RuntimeError("halo2: constraint system not satisfied at verify")

    return {
        "num_rows": num_rows,
        "logrows": logrows,
        "settings_s": round(settings_s, 4),
        "compile_s": round(compile_s, 4),
        "setup_s": round(setup_s, 4),
        "witness_s": round(witness_s, 4),
        "prove_s": round(prove_s, 4),
        "verify_ms": round(verify_ms, 3),
        "proof_kb": round(proof_kb, 3),
        "peak_rss_mb": round(peak, 2),
        "scale_bits": int(scale_bits) if scale_bits is not None else 0,
        "verified": True,
        "settings_path": str(settings_path),
        "proof_path": str(proof_path),
        "witness_path": str(witness_path),
    }


def measure_prepared(
    model,
    inputs: list,
    work_dir: Path,
    *,
    input_names: list[str],
    output_names: list[str] | None = None,
    name: str = "prepared",
    train_epochs: int = 0,   # kept for signature parity; ignored (model is given)
    calibrate: bool = False,
    drop_proving_key: bool = True,
    input_visibility: str = "private",
    output_visibility: str = "public",
    input_scale: int = 7,
) -> RunResult:
    """Measure an already-built model with an explicit input list.

    This is the entry point used by ``zk.binding`` for the bare/bound toggle:
    the caller decides how many graph inputs there are and what the public
    instance contains.  Every number still comes from ``_attempt``.
    """
    import ezkl
    import torch

    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = work_dir / "model.onnx"
    torch.onnx.export(
        model.eval(),
        tuple(inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names or ["output"],
        opset_version=17,
        dynamo=False,
    )
    data_path = work_dir / "input.json"
    # EZKL 23.0.5: one entry per graph input, in graph order.
    data_path.write_text(json.dumps({"input_data": [t.detach().cpu().reshape(-1).tolist()
                                                    for t in inputs]}))

    errors: list[str] = []
    out: dict | None = None
    for attempt, scale in enumerate((input_scale, 12, 14)):
        args = ezkl.PyRunArgs()
        args.input_visibility = input_visibility
        args.output_visibility = output_visibility
        args.param_visibility = "fixed"
        args.input_scale = int(scale)
        args.param_scale = int(scale)
        args.scale_rebase_multiplier = 1
        try:
            out = _attempt(
                onnx_path=onnx_path,
                data_path=data_path,
                work_dir=work_dir,
                logrows_hint=True,
                scale_bits=scale,
                calibrate=calibrate,
                drop_proving_key=drop_proving_key,
                tag="" if attempt == 0 else f"s{scale}",
                run_args=args,
            )
            break
        except Exception as exc:  # noqa: BLE001 - try the next scale, keep the trace
            errors.append(f"scale={scale}: {type(exc).__name__}: {exc}")
            print(f"warn: attempt {attempt + 1} failed ({errors[-1]})", file=sys.stderr)
    if out is None:
        raise RuntimeError(" | ".join(errors))
    return RunResult(
        name=name,
        batch=1,
        num_rows=int(out["num_rows"]),
        logrows=int(out["logrows"]),
        n_params=int(sum(p.numel() for p in model.parameters())),
        input_dim=int(inputs[0].numel()),
        settings_s=float(out["settings_s"]),
        compile_s=float(out["compile_s"]),
        setup_s=float(out["setup_s"]),
        witness_s=float(out["witness_s"]),
        prove_s=float(out["prove_s"]),
        verify_ms=float(out["verify_ms"]),
        proof_kb=float(out["proof_kb"]),
        peak_rss_mb=float(out["peak_rss_mb"]),
        verified=bool(out["verified"]),
        work_dir=str(work_dir),
        scale_bits=int(out["scale_bits"]),
        attempts=1,
        status="ok" if out["verified"] else "verify_failed",
        error="",
    )


def measure(
    bundle: WorkloadBundle,
    work_dir: Path,
    batch: int = 1,
    train_epochs: int = 5,
    seed: int = 0,
    calibrate: bool = True,
    drop_proving_key: bool = True,
) -> RunResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    model = train_bundle(bundle, epochs=train_epochs, seed=seed)
    sample = sample_input(bundle, batch=batch)
    onnx_path = work_dir / "model.onnx"
    export_onnx(model, sample, onnx_path)

    data_path = work_dir / "input.json"
    arr = sample.detach().cpu().numpy()
    data_path.write_text(json.dumps({"input_data": [arr.reshape(-1).tolist()]}))

    n_params = int(sum(p.numel() for p in model.parameters()))
    input_dim = int(arr.size)

    errors: list[str] = []
    out: dict | None = None
    attempt_no = 0
    for scale_bits in SCALE_ATTEMPTS:
        attempt_no += 1
        tag = "" if scale_bits is None else f"s{scale_bits}"
        try:
            out = _attempt(
                onnx_path=onnx_path,
                data_path=data_path,
                work_dir=work_dir,
                logrows_hint=True,
                scale_bits=scale_bits,
                calibrate=calibrate,
                drop_proving_key=drop_proving_key,
                tag=tag,
            )
            break
        except Exception as exc:  # noqa: BLE001 - try the next scale, keep the trace
            errors.append(f"scale={scale_bits or 'auto'}: {type(exc).__name__}: {exc}")
            print(f"warn: attempt {attempt_no} failed ({errors[-1]})", file=sys.stderr)

    if out is None:
        raise RuntimeError(" | ".join(errors))

    return RunResult(
        name=bundle.name,
        batch=batch,
        num_rows=int(out["num_rows"]),
        logrows=int(out["logrows"]),
        n_params=n_params,
        input_dim=input_dim,
        settings_s=float(out["settings_s"]),
        compile_s=float(out["compile_s"]),
        setup_s=float(out["setup_s"]),
        witness_s=float(out["witness_s"]),
        prove_s=float(out["prove_s"]),
        verify_ms=float(out["verify_ms"]),
        proof_kb=float(out["proof_kb"]),
        peak_rss_mb=float(out["peak_rss_mb"]),
        verified=bool(out["verified"]),
        work_dir=str(work_dir),
        scale_bits=int(out["scale_bits"]),
        attempts=attempt_no,
        status="ok" if out["verified"] else "verify_failed",
        error="; ".join(errors),
    )


def result_dict(r: RunResult) -> dict:
    return asdict(r)


# --------------------------------------------------------------------------
# subprocess entry point used by zk.scaling (isolates peak RSS per config)
# --------------------------------------------------------------------------
def _cli() -> int:
    import argparse

    from zk import workloads as W

    def _binding_configs():
        from zk.binding import CONFIGS
        return set(CONFIGS)

    ap = argparse.ArgumentParser(description="bench one workload config")
    ap.add_argument("--config", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--keep-keys", action="store_true",
                    help="keep the multi-GB proving key (default: delete after verify)")
    ap.add_argument("--binding", choices=("none", "bare", "bound"), default="none",
                    help="none: plain workload; bare/bound: synthetic binding-layer harness")
    a = ap.parse_args()

    if a.binding != "none":
        if a.config not in _binding_configs():
            print(f"--binding {a.binding} only applies to {sorted(_binding_configs())}",
                  file=sys.stderr)
            return 2
        from zk.binding import _measure_bare_or_bound

        row = _measure_bare_or_bound(a.config, a.binding, Path(a.work_dir))
        print(json.dumps(row))
        return 0 if row.get("verified") else 1

    factory = W.FACTORIES.get(a.config)
    if factory is None:
        print(f"unknown config {a.config!r}; known: {sorted(W.FACTORIES)}", file=sys.stderr)
        return 2
    try:
        res = measure(
            factory(),
            Path(a.work_dir),
            batch=a.batch,
            train_epochs=a.epochs,
            seed=a.seed,
            calibrate=not a.no_calibrate,
            drop_proving_key=not a.keep_keys,
        )
        print(json.dumps(result_dict(res)))
        return 0
    except Exception as exc:  # noqa: BLE001 - caller records the failure
        row = {
            "name": a.config,
            "batch": a.batch,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "work_dir": a.work_dir,
            "system": f"{platform.system()} {platform.machine()}",
        }
        print(json.dumps(row))
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
