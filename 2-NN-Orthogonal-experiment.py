#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D  # ✅ NEW

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  
    "mathtext.fontset": "stix",         
    "pdf.fonttype": 42,                 
    "ps.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser(description="2-layer NN: multi-lr_max diagonal curves with colorbar (arbitrary steps)")
    p.add_argument("--d", type=int, default=1000, help="Input dimension (d)")
    p.add_argument("--h", type=int, default=1000, help="Hidden width (h)")
    p.add_argument("--n", type=int, default=1000, help="Train set size (n)")
    p.add_argument("--test-size", type=int, default=500, help="Test set size")

    # new sweep args
    p.add_argument("--lr-max-start", type=int, default=27000, help="Start lr_max (inclusive)")
    p.add_argument("--lr-max-end", type=int, default=63000, help="End lr_max (inclusive)")
    p.add_argument("--lr-max-step", type=int, default=2000, help="Step for lr_max sweep")

    # NEW: arbitrary steps list, e.g. "1,2,4,8"
    p.add_argument("--steps-list", type=str, default="1,2",
                   help="Comma-separated list of GD steps to evaluate, e.g. '1,2,4,8'")


    p.add_argument("--outdir", type=str, default="_", help="Output directory")
    return p.parse_args()


def orthonormal_columns(rows, cols, *, device, dtype=None):
    """
    Return Q (rows x cols) with Q^T Q = I_cols. Requires rows >= cols.
    Construct via QR decomposition and stabilize column signs.
    """
    if rows < cols:
        raise ValueError(f"Cannot construct {rows}x{cols} with orthonormal columns (need rows >= cols).")
    A = torch.randn(rows, cols, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A, mode='reduced')
    diag = torch.diag(R)
    signs = torch.sign(diag)
    signs[signs == 0] = 1
    Q = Q * signs
    return Q


def train_step(W1, W2, X, y, a, h, lr1, lr2):
    n = X.size(0)

    W1_ = W1.detach().clone().requires_grad_(True)
    W2_ = W2.detach().clone().requires_grad_(True)

    pred = (1.0 / h) * X @ W1_ @ W2_ @ a
    loss = (0.5 / n) * torch.norm(pred - y, p='fro') ** 2
    loss.backward()

    with torch.no_grad():
        W1_new = W1_ - float(lr1) * W1_.grad
        W2_new = W2_ - float(lr2) * W2_.grad

    return W1_new.detach(), W2_new.detach(), loss.detach()


def compute_curves_for_one_lrmax(lr_max, *, d, h, n, test_size, device, seeds, steps_list):
    """
    For a given lr_max, compute the diagonal curve (eta1+eta2=lr_max) for ALL steps in steps_list.
    Returns:
      xs: (num_points,)
      losses_by_step: dict {step: (num_points,) average test loss}
    Uses exactly the same data generation + test loss as your code.
    """
    # same grid logic as your code
    step_val = max(1, lr_max // 20) if lr_max > 0 else 1
    N = int(lr_max / step_val)  # your code
    xs = np.array([i * step_val for i in range(1, N)], dtype=float)  # eta1
    lr2s = np.array([lr_max - x for x in xs], dtype=float)           # eta2

    steps_list = sorted(set(int(s) for s in steps_list))
    if any(s <= 0 for s in steps_list):
        raise ValueError("All steps in steps_list must be positive integers.")

    # accumulate losses
    sum_losses = {s: np.zeros_like(xs, dtype=float) for s in steps_list}

    for seed in seeds:
        torch.manual_seed(seed)

        rho = 0.0
        #rho = 0.001

        # Test inputs (no orthogonality constraint required)
        x = torch.randn(test_size, d, device=device)

        # Orthogonal init + orthogonal X scaled
        W1_init = orthonormal_columns(d, h, device=device, dtype=torch.float32)
        W2_init = orthonormal_columns(h, h, device=device, dtype=torch.float32)

        Qx = orthonormal_columns(n, d, device=device, dtype=torch.float32)
        X = torch.sqrt(torch.tensor(float(n), device=device)) * Qx
        Qb = orthonormal_columns(d, h, device=device, dtype=torch.float32)

        # Same as your code
        xi = torch.normal(mean=0, std=np.sqrt(rho), size=(n, h), device=device)  # n x h
        a = torch.eye(h, device=device)                                          # h x h
        beta = torch.sqrt(torch.tensor(float(1 / d), device=device)) * Qb        # d x h
        y = X @ beta + xi                                                        # n x h

        with torch.no_grad():
            xb = x @ beta  # (test_size x h), reused for all lr pairs in this seed

        for k in range(xs.shape[0]):
            lr1 = float(xs[k])
            lr2 = float(lr2s[k])

            W1 = W1_init.clone()
            W2 = W2_init.clone()

            # run GD and record at requested steps
            t = 0
            for target_step in steps_list:
                while t < target_step:
                    W1, W2, _ = train_step(W1, W2, X, y, a, h, lr1, lr2)
                    t += 1

                with torch.no_grad():
                    pred_test = (1.0 / h) * x @ W1 @ W2 @ a
                    resid = pred_test - xb
                    test_loss = torch.norm(resid, p='fro') ** 2 / x.size(0)

                sum_losses[target_step][k] += float(test_loss)

    avg_losses = {s: sum_losses[s] / len(seeds) for s in steps_list}
    return xs, avg_losses


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    d = int(args.d)
    h = int(args.h)
    n = int(args.n)
    test_size = int(args.test_size)

    # Feasibility constraints for exact equalities: h <= d <= n
    if h > d:
        raise ValueError(f"h ({h}) must be <= d ({d}) to have W1^T W1 = I_h.")
    if d > n:
        raise ValueError(f"d ({d}) must be <= n ({n}) to have X^T X = n I_d.")

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

    # parse steps-list
    steps_list = [int(s.strip()) for s in args.steps_list.split(",") if s.strip() != ""]
    if len(steps_list) == 0:
        raise ValueError("--steps-list is empty.")
    if any(s <= 0 for s in steps_list):
        raise ValueError("All steps in --steps-list must be positive integers.")
    steps_list = sorted(set(steps_list))

    os.makedirs(args.outdir, exist_ok=True)

    # keep your plot sizes
    plt.rcParams.update(
        {"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 20, "legend.fontsize": 14}
    )

    seeds = range(2020, 2025)



    base = plt.get_cmap("plasma")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "plasma_0_85",
        base(np.linspace(0.0, 0.85, 256))
    )

    norm = mpl.colors.Normalize(vmin=min(lr_max_values), vmax=max(lr_max_values))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    
    legend_handles = [
        Line2D(
            [0], [0],
            marker='*', linestyle='None',
            markersize=18,
            markerfacecolor='black', markeredgecolor='black',
            label=r"$\eta_1=\eta_2$"
        ),
    ]

    # For each requested step, create a separate figure
    figs = {}
    axes = {}
    for s in steps_list:
        figs[s], axes[s] = plt.subplots(figsize=(8, 6))

    # Sweep lr_max and add curves to each step-figure
    for lr_max in lr_max_values:
        xs, avg_losses = compute_curves_for_one_lrmax(
            lr_max,
            d=d, h=h, n=n, test_size=test_size,
            device=device,
            seeds=seeds,
            steps_list=steps_list
        )

        color = cmap(norm(lr_max))
        
        x_mid = 0.5 * lr_max
        idx_mid = int(np.argmin(np.abs(xs - x_mid)))

        for s in steps_list:
            axes[s].plot(xs, avg_losses[s], linestyle='-', linewidth=4, color=color)

            
            axes[s].plot(
                xs[idx_mid], avg_losses[s][idx_mid],
                marker='*', markersize=15,
                markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.8,
                linestyle='None', zorder=10
            )

    # style + colorbar + save
    for s in steps_list:
        ax = axes[s]
        fig = figs[s]

        ax.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.set_ylabel(r"Test Loss", fontsize=40)
        ax.set_xlabel(r" $\eta_1$", fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)

        cbar = fig.colorbar(sm, ax=ax)
        
        cbar.set_label(r" $\eta_1+\eta_2$")

        
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=20)

        out_pdf = os.path.join(
            args.outdir,
            f"2-NN-diagonal-multicurve_rho_{0.001}_step_{s}_n_{n}_d_{d}_h_{h}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}.pdf"
        )
        out_png = os.path.join(
            args.outdir,
            f"2-NN-diagonal-multicurve_rho_{0.001}_step_{s}_n_{n}_d_{d}_h_{h}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}.png"
        )

        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
