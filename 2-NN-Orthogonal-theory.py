#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch  
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D  

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "mathtext.fontset": "stix",        
    "pdf.fonttype": 42,                 
    "ps.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser(description="2-layer NN: plot diagonal curves for multiple lr_max with colorbar")
    p.add_argument("--d", type=int, default=1000, help="Input dimension (d) (kept for filename compatibility)")
    p.add_argument("--h", type=int, default=1000, help="Hidden width (h)")
    p.add_argument("--n", type=int, default=1000, help="Train set size (n) (kept for filename compatibility)")

    # produce two figures by default (N_steps=1 and 2)
    p.add_argument("--steps-list", type=str, default="1,2", help="Comma-separated list of N_steps to plot, e.g. '1,2'")

    # lr_max sweep config
    p.add_argument("--lr-max-start", type=int, default=27000, help="Start lr_max (inclusive)")
    p.add_argument("--lr-max-end", type=int, default=63000, help="End lr_max (inclusive)")
    p.add_argument("--lr-max-step", type=int, default=2000, help="Step for lr_max sweep")

    # output directory
    p.add_argument("--outdir", type=str, default="_", help="Output directory")
    return p.parse_args()


def theoretical_test_loss(lr1, lr2, h, N_steps):
    # EXACT same formulas as your original code.
    h = float(h)
    lr1 = float(lr1)
    lr2 = float(lr2)

    if N_steps == 2:
        return (
            1.0 / h * (1.0 + lr1 * lr2 / (h ** 3)) ** 4
            + 16.0 * lr1 ** 2 * lr2 ** 2 / (h ** 7)
            + (2.0 * (lr1 + lr2) * (lr1 * lr2 + h ** 3) / (h ** 5) - 1.0) ** 2
            + (1.0 + lr1 * lr2 / (h ** 3)) ** 2 * 8.0 * lr1 * lr2 / (h ** 5)
        )
    elif N_steps == 1:
        return (
            lr1 ** 2 / (h ** 4)
            + lr2 ** 2 / (h ** 4)
            + 2.0 * lr1 * lr2 / (h ** 4)
            + lr1 ** 2 * lr2 ** 2 / (h ** 7)
            - 2.0 * lr1 / (h ** 2)
            - 2.0 * lr2 / (h ** 2)
            + 1.0 / h
            + 1.0
        )
    else:
        raise ValueError("Only N_steps=1 or 2 are supported.")


def build_diagonal_curve(lr_max, h, N_steps):
    # EXACT same grid logic as your original code
    step_val = max(1, lr_max // 20) if lr_max > 0 else 1

    const = lr_max
    N = int(const / step_val)

    xs, L = [], []
    for i in range(1, N):
        j = N - i
        lr1 = i * step_val
        lr2 = j * step_val
        xs.append(lr1)
        L.append(theoretical_test_loss(lr1, lr2, h, N_steps))

    return np.array(xs, dtype=float), np.array(L, dtype=float), step_val


def main():
    args = parse_args()

    # keep your plotting rcParams update (do not change sizes)
    plt.rcParams.update(
        {"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 20, "legend.fontsize": 14}
    )

    d = int(args.d)
    h = int(args.h)
    n = int(args.n)

    steps_list = [int(s.strip()) for s in args.steps_list.split(",") if s.strip() != ""]
    for s in steps_list:
        if s not in (1, 2):
            raise ValueError(f"steps-list contains {s}, but only 1 and 2 are supported.")

    lr_max_start = int(args.lr_max_start)
    lr_max_end = int(args.lr_max_end)
    lr_max_step = int(args.lr_max_step)

    if lr_max_step <= 0:
        raise ValueError("--lr-max-step must be positive.")
    if lr_max_end < lr_max_start:
        raise ValueError("--lr-max-end must be >= --lr-max-start.")

    lr_max_values = list(range(lr_max_start, lr_max_end + 1, lr_max_step))
    if len(lr_max_values) == 0:
        raise ValueError("No lr_max values to plot; check your start/end/step.")

    os.makedirs(args.outdir, exist_ok=True)

    # colormap + normalization for colorbar
    # cmap = plt.get_cmap("viridis")
    # norm = mpl.colors.Normalize(vmin=min(lr_max_values), vmax=max(lr_max_values))
    # sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    base = plt.get_cmap("plasma")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "plasma_0_85",
        base(np.linspace(0.0, 0.85, 256))
    )

    norm = mpl.colors.Normalize(vmin=min(lr_max_values), vmax=max(lr_max_values))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ✅ NEW: legend handle for eta1=eta2 (same as your previous merged code)
    legend_handles = [
        Line2D(
            [0], [0],
            marker='*', linestyle='None',
            markersize=18,
            markerfacecolor='black', markeredgecolor='black',
            label=r"$\eta_1=\eta_2$"
        ),
    ]

    for N_steps in steps_list:
        fig, ax = plt.subplots(figsize=(8, 6))

        for lr_max in lr_max_values:
            xs, ys, _ = build_diagonal_curve(lr_max=lr_max, h=h, N_steps=N_steps)
            color = cmap(norm(lr_max))

            ax.plot(
                xs, ys,
                linestyle='-',
                linewidth=4,
                color=color,
            )

           
            x_mid = 0.5 * lr_max
            idx_mid = int(np.argmin(np.abs(xs - x_mid)))
            ax.plot(
                xs[idx_mid], ys[idx_mid],
                marker='*', markersize=15,
                markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.8,
                linestyle='None', zorder=10
            )

        ax.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.set_ylabel(r"Test Loss", fontsize=40)
        ax.set_xlabel(r" $\eta_1$", fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)

       
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r" $\eta_1+\eta_2$")

       
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=20)

        out_pdf = os.path.join(
            args.outdir,
            f"2-NN-theoretical-diagonal-multicurve_step_{N_steps}_n_{n}_d_{d}_h_{h}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}.pdf"
        )
        out_png = os.path.join(
            args.outdir,
            f"2-NN-theoretical-diagonal-multicurve_step_{N_steps}_n_{n}_d_{d}_h_{h}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}.png"
        )

        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
