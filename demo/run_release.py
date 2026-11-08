"""Terminal demo: one full DP-zkML release, then the attacks that must fail.

    python -m demo.run_release              # full run with the negative tests
    python -m demo.run_release --no-attacks # happy path only
    python -m demo.run_release --plain      # no rich, plain text fallback

Screen-recording notes live in demo/README.md. Every check printed here is a
real re-derivation; the one thing NOT implemented (the proof backend) is
printed as NOT BOUND rather than glossed over.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from demo.pipeline import (
    C_Q,
    DIM,
    EPS_CAP,
    N_RECORDS,
    SIGMA,
    ReleaseSession,
    verify_report,
)
from quant.reference_query import synthetic_records

try:  # rich is demo-only; fall back to plain output when it is absent
    from rich.box import SIMPLE
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console = Console()
    HAVE_RICH = True
except ModuleNotFoundError:  # pragma: no cover
    _console = None
    HAVE_RICH = False


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def rule(title: str) -> None:
    if HAVE_RICH:
        _console.rule(f"[bold]{title}[/bold]")
    else:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def line(text: str) -> None:
    print(text)


def kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    if HAVE_RICH:
        t = Table(title=title, box=SIMPLE, show_header=False, pad_edge=False)
        t.add_column(style="dim", no_wrap=True)
        t.add_column(style="bold")
        for k, v in rows:
            t.add_row(k, v)
        _console.print(t)
    else:
        print(f"\n-- {title} --")
        for k, v in rows:
            print(f"  {k:<28} {v}")


def ok_panel(msg: str, detail: str = "") -> None:
    if HAVE_RICH:
        _console.print(Panel(f"[bold green]{msg}[/bold green]"
                             + (f"\n[dim]{detail}[/dim]" if detail else ""),
                             border_style="green", box=SIMPLE, expand=False))
    else:
        print(f"[ OK ] {msg}" + (f"  ({detail})" if detail else ""))


def bad_panel(msg: str, detail: str = "") -> None:
    """A negative test firing: red border, unambiguous BLOCKED verdict."""
    if HAVE_RICH:
        _console.print(Panel(f"[bold white]{msg}[/bold white]  ->  [bold red]BLOCKED[/bold red]"
                             + (f"\n[dim]{detail}[/dim]" if detail else ""),
                             border_style="red", box=SIMPLE, expand=False))
    else:
        print(f"[BLOCKED] {msg}" + (f"  ({detail})" if detail else ""))


def spark(values: np.ndarray) -> str:
    """Compact inline histogram so the noise is visible in a terminal."""
    blocks = " ▁▂▃▄▅▆▇█"
    v = np.abs(np.asarray(values, dtype=float))
    if v.max() <= 0:
        return blocks[0] * len(v)
    idx = np.round(v / v.max() * (len(blocks) - 2)).astype(int) + 1
    return "".join(blocks[i] for i in idx)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DP-zkML single-release demo")
    ap.add_argument("--no-attacks", action="store_true", help="skip negative tests")
    ap.add_argument("--plain", action="store_true", help="force plain text output")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--cq", type=float, default=C_Q)
    ap.add_argument("--records", type=int, default=N_RECORDS)
    args = ap.parse_args(argv)

    global HAVE_RICH
    if args.plain:
        HAVE_RICH = False

    session = ReleaseSession(sigma=args.sigma, c_q=args.cq, dim=DIM,
                             eps_cap=EPS_CAP)

    rule("DP-zkML  |  verifiable differentially-private inference")
    print("  Outsourced inference, three guarantees, one release protocol.")
    print(f"  d={DIM}  records={args.records}  C_Q={args.cq:g}  "
          f"sigma={args.sigma:g}  eps_cap={EPS_CAP:g}  delta=1e-5")

    # spread 0.5 keeps the record norms near C_Q, so clipping is selective
    # rather than saturating every row -- that is what the mechanism looks like
    records = synthetic_records(args.records, DIM, seed=0, spread=0.5)
    rel = session.release(records=records)

    # 1 -------------------------------------------------------------- input
    rule("1/5  Input  --  data custodian commits the record set")
    kv_table("public commitments", [
        ("C_D    (dataset)", f"{rel.commitments['C_D'][:32]}..."),
        ("C_theta (model)", f"{rel.commitments['C_theta'][:32]}..."),
        ("C_rho   (seed)", f"{rel.commitments['C_rho'][:32]}...  [hidden witness]"),
        ("records", f"{records.shape[0]} x {records.shape[1]} (synthetic, seed 0)"),
    ])

    # 2 ------------------------------------------------------------ compute
    rule("2/5  Compute  --  quantize, clip, aggregate")
    kv_table("sensitivity certificate", [
        ("raw max ||v_theta(r)||_2", f"{rel.max_raw_norm:.2f}"),
        ("post-clip max ||v||_2", f"{rel.max_clipped_norm:.2f}"),
        ("clip radius C_Q", f"{rel.c_q:.1f}"),
        ("records clipped", f"{rel.clipped} / {records.shape[0]}"),
        ("replacement sensitivity", f"2 * C_Q = {rel.sensitivity:.1f}"),
        ("sufficient field bound", f"p > {rel.field_bound:,}"),
    ])
    print(f"\n  z = g_theta(D)  first 8 dims: {rel.z[:8].tolist()}")
    print(f"  |z|_1 = {int(np.abs(rel.z).sum()):,}   (no noise yet -- never published)")

    # 3 --------------------------------------------------------------- noise
    rule("3/5  Noise  --  exact-support discrete Gaussian")
    kv_table("sampler", [
        ("sigma", f"{rel.sigma:g}"),
        ("support", f"[-{int(np.abs(rel.pmf_support).max())}, "
                    f"{int(np.abs(rel.pmf_support).max())}]"),
        ("pmf sums to", f"{rel.pmf_probs.sum():.12f}"),
        ("excluded tail mass", f"{rel.tail_mass:.2e}  (bound 1e-12)"),
    ])
    print(f"\n  eta  first 16: {spark(rel.eta)}   sum={int(rel.eta.sum()):+d}")
    print(f"  seed rho stays a hidden witness: only its commitment C_rho is public.")

    # 4 ------------------------------------------------------------- release
    rule("4/5  Output  --  publish y and advance the budget")
    kv_table("release", [
        ("y = Pi(z + eta)", f"{rel.y[:8].tolist()} ..."),
        ("C_y", f"{rel.commitments['C_y'][:32]}..."),
        ("epsilon (tighter)", f"{rel.eps_point:.4f}"),
        ("epsilon (enforced)", f"{rel.eps_upper:.4f}   <= cap {rel.eps_cap:g}"),
        ("releases used", f"{rel.releases_used} of {rel.releases_allowed} at eps<={rel.eps_cap:g}"),
        ("ledger root", f"{rel.prev_root[:12]}... -> {rel.root[:12]}..."),
    ])

    # 5 -------------------------------------------------------------- verify
    rule("5/5  Verify  --  re-derive every public check")
    checks = session.verify(rel)
    for c in checks:
        mark = "PASS" if c.ok else ("FAIL" if c.ok is False else "OPEN")
        colour = {"PASS": "\033[32m", "FAIL": "\033[31m", "OPEN": "\033[33m"}[mark]
        paint = sys.stdout.isatty() and not HAVE_RICH
        reset = "\033[0m" if paint else ""
        lead = colour if paint else ""
        print(f"  {lead}[{mark}]{reset} {c.name:<34} {c.detail}")
    n_ok, n_bad, n_open = verify_report(checks)
    print(f"\n  {n_ok} passed, {n_bad} failed, {n_open} open")

    if args.no_attacks:
        return 0 if n_bad == 0 else 1

    # ------------------------------------------------------ negative tests
    rule("Negative tests  --  the attacks that must fail")

    msg = session.attack_replay(rel)
    bad_panel(f"replay {rel.release_id}", msg)

    msg = session.attack_stale(rel)
    bad_panel("stale predecessor", msg)

    msg = session.attack_second_child()
    bad_panel("second successor of one state", msg)

    tampered = rel.y.copy()
    tampered[0] += 1
    t_checks = session.verify(rel, artifact_y=tampered)
    failed = [c.name for c in t_checks if c.ok is False]
    bad_panel("tampered artifact", f"{', '.join(failed)} failed")

    # ------------------------------------------------------------- summary
    rule("Summary")
    b = session.budget()
    ok_panel("release protocol consistent",
             f"eps={b['eps_upper']:.4f}/{b['eps_cap']:g}   "
             f"releases={b['releases_used']}/{b['releases_allowed']}   "
             f"root={b['root'][:12]}...")
    print("  proof backend: NOT BOUND  (Track D -- the part this grant funds)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
