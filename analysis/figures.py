"""Regenerate analytical Figs 3–9 into results/ (paper Sec. VII–VIII)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d

from analysis.accounting import rdp_epsilon
from analysis.constraints import constraint_proxy, proving_work_index
from analysis.utility import calibrated_utility
from analysis.verification import verification_density, verification_median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _save(fig: plt.Figure, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_rdp_composition() -> Path:
    """Fig. 3: analytical RDP-to-DP bounds vs release count T."""
    T = np.arange(1, 41)
    sigmas = [2, 4, 8, 16]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for sigma in sigmas:
        eps = [rdp_epsilon(sigma, int(t)) for t in T]
        ax.plot(T, eps, label=rf"$\sigma/\Delta={sigma}$")
    ax.axhline(8.0, color="0.35", ls="--", lw=1.0, label=r"$\varepsilon_{\mathrm{max}}=8$")
    ax.set_xlabel("Releases $T$ (no subsampling)")
    ax.set_ylabel("RDP-derived $\\varepsilon$ bound (analytical)")
    ax.set_ylim(0, 22)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Fig. 3 — Analytical repeated-release privacy bound")
    return _save(fig, "fig3_rdp_composition.png")


def fig4_calibrated_utility() -> Path:
    """Fig. 4: synthetic utility vs ε with RDP-calibrated σ."""
    eps_grid = np.linspace(0.5, 10.0, 40)
    capacities = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in capacities:
        u = [calibrated_utility(c=float(c), eps=float(e), T=1) for e in eps_grid]
        ax.plot(eps_grid, u, label=f"Capacity index {c}")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ ($T=1$)")
    ax.set_ylabel("Synthetic utility score (analytical)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Fig. 4 — Calibrated synthetic utility proxy")
    return _save(fig, "fig4_calibrated_utility.png")


def fig5_constraint_shares() -> Path:
    """Fig. 5: stacked assumed constraint shares for three (d,w) scenarios."""
    scenarios = [(4, 64), (8, 128), (16, 256)]
    labels = [f"{d}/{w}" for d, w in scenarios]
    keys = ["RML", "Rrange", "Rsample", "Raccount", "Rbind"]
    pretty = [
        "ML arithmetic",
        "Range checks",
        "Sampler",
        "Accountant",
        "Binding / state",
    ]
    shares = {k: [] for k in keys}
    for d, w in scenarios:
        out = constraint_proxy(d=d, w=w)
        total = out["R"]
        for k in keys:
            shares[k].append(100.0 * out[k] / total)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for k, name, color in zip(keys, pretty, colors):
        vals = np.array(shares[k])
        ax.bar(x, vals, bottom=bottom, label=name, color=color, width=0.55)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Depth / width")
    ax.set_ylabel("Assumed constraint share (%) (analytical)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Fig. 5 — Assumed constraint shares")
    return _save(fig, "fig5_constraint_shares.png")


def fig6_proving_work_surface() -> Path:
    """Fig. 6: synthetic proving-work surface + contour of log10 IP."""
    depths = np.arange(2, 21)
    widths = np.arange(16, 401, 8)
    D, W = np.meshgrid(depths, widths, indexing="ij")
    IP = np.zeros_like(D, dtype=float)
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            r = constraint_proxy(d=int(d), w=int(w))["R"]
            IP[i, j] = proving_work_index(r)
    log_ip = np.log10(IP)

    fig = plt.figure(figsize=(10.5, 4.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(D, W, log_ip, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_xlabel("Depth $d$")
    ax1.set_ylabel("Width $w$")
    ax1.set_zlabel(r"$\log_{10} I_P$ (analytical)")
    ax1.set_title("(a) Synthetic proving-work surface")

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(D, W, IP, levels=[0.25, 1, 4, 16, 64, 256], colors="k", linewidths=0.8)
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%g")
    pcm = ax2.contourf(D, W, log_ip, levels=20, cmap=cm.viridis)
    fig.colorbar(pcm, ax=ax2, label=r"$\log_{10} I_P$ (analytical)")
    ax2.set_xlabel("Depth $d$")
    ax2.set_ylabel("Width $w$")
    ax2.set_title("(b) Matching contour projection")
    fig.suptitle("Fig. 6 — Synthetic proving-work scaling", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig6_proving_work.png")


def fig7_overhead_heatmap() -> Path:
    """Fig. 7: added-constraint overhead (%) at fixed depth d=8."""
    widths = [32, 64, 128, 256, 512]
    ms = [16, 64, 128, 256, 512]
    heat = np.zeros((len(ms), len(widths)))
    for i, m in enumerate(ms):
        for j, w in enumerate(widths):
            out = constraint_proxy(d=8, w=w, m=m)
            heat[i, j] = out["overhead_pct"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms)
    ax.set_xlabel("Width $w$ at depth $d=8$")
    ax.set_ylabel("Released coordinates $m$")
    for i in range(len(ms)):
        for j in range(len(widths)):
            ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Overhead (%) (analytical)")
    ax.set_title("Fig. 7 — Added-constraint overhead")
    return _save(fig, "fig7_overhead_heatmap.png")


def fig8_verification_violins() -> Path:
    """Fig. 8: analytical log-normal densities of verification-work index."""
    ss = [8, 32, 128, 512]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    positions = []
    for idx, s in enumerate(ss):
        xs, dens = verification_density(s=s, n=400)
        # vertical violin: map density to half-width around categorical x
        x0 = idx
        positions.append(x0)
        half = 0.35 * dens / dens.max()
        ax.fill_betweenx(xs, x0 - half, x0 + half, alpha=0.55, color="#4C72B0")
        ax.plot(x0, verification_median(s), "o", color="white", ms=5, mec="0.2")
    ax.set_xticks(positions)
    ax.set_xticklabels(ss)
    ax.set_xlabel("Public field elements $s$")
    ax.set_ylabel("Synthetic verification-work index (analytical)")
    ax.set_title("Fig. 8 — Analytical verification-work densities")
    return _save(fig, "fig8_verification_densities.png")


def fig9_ablation() -> Path:
    """Fig. 9: one-component omissions vs full-relation proving work."""
    base = constraint_proxy(d=8, w=128)
    r_full = base["R"]
    ip_full = proving_work_index(r_full)

    variants = [
        ("Full relation", 0),
        ("No range checks", base["Rrange"]),
        ("No sampler", base["Rsample"]),
        ("No accountant", base["Raccount"]),
        ("No binding/state", base["Rbind"]),
    ]
    xs, ys, labels = [], [], []
    for name, drop in variants:
        r = r_full - drop
        overhead = 100.0 * (r - base["RML"]) / base["RML"]
        ratio = proving_work_index(r) / ip_full
        xs.append(ratio)
        ys.append(overhead)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(xs, ys, s=40, color="#4C72B0")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Proving work / full-relation work (analytical)")
    ax.set_ylabel("Overhead (%) (analytical)")
    ax.set_xlim(0.82, 1.02)
    ax.set_ylim(-5, 650)
    ax.set_title("Fig. 9 — One-component omissions (diagnostic)")
    return _save(fig, "fig9_ablation.png")


def main() -> None:
    paths = [
        fig3_rdp_composition(),
        fig4_calibrated_utility(),
        fig5_constraint_shares(),
        fig6_proving_work_surface(),
        fig7_overhead_heatmap(),
        fig8_verification_violins(),
        fig9_ablation(),
    ]
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
