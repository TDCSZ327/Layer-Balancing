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
    p = argparse.ArgumentParser(
        description="3-layer linear NN: sweep lr_max and plot diagonal curves with a lr_max colorbar (arbitrary steps)"
    )
    p.add_argument("--d", type=int, default=1000, help="Input dimension (d)")
    p.add_argument("--h", type=int, default=1000, help="Hidden width (h)")
    p.add_argument("--n", type=int, default=1000, help="Train set size (n)")
    p.add_argument("--test-size", type=int, default=500, help="Test set size")

    # sweep lr_max
    p.add_argument("--lr-max-start", type=int, default=30, help="Start lr_max (inclusive)")
    p.add_argument("--lr-max-end", type=int, default=100, help="End lr_max (inclusive)")
    p.add_argument("--lr-max-step", type=int, default=5, help="Step for lr_max sweep")

    p.add_argument("--outdir", type=str, default="_", help="Output directory")

    # NEW: arbitrary steps
    p.add_argument("--steps-list", type=str, default="1,2",
                   help="Comma-separated list of GD steps to evaluate, e.g. '1,2,4,8'")
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

    inv_sqrt_h = 1.0 / torch.sqrt(torch.tensor(h, dtype=W1.dtype, device=W1.device))

    pred = inv_sqrt_h * X @ W1_ @ W2_ @ a
    loss = (0.5 / n) * torch.norm(pred - y, p='fro') ** 2
    loss.backward()

    with torch.no_grad():
        W1_new = W1_ - float(lr1) * W1_.grad
        W2_new = W2_ - float(lr2) * W2_.grad

    return W1_new.detach(), W2_new.detach(), loss.detach()


def compute_curves_for_one_lrmax(lr_max, *, d, h, n, test_size, device, seeds, steps_list):
    """
    For a given lr_max, compute the diagonal curve (indices i+j=N, same as your code)
    for ALL steps in steps_list, using exactly the same data generation + test loss as your code.

    Returns:
      xs: (num_points,)
      losses_by_step: dict {step: (num_points,) average test loss}
    """
    step_val = max(1, lr_max // 20) if lr_max > 0 else 1
    N = int(lr_max / step_val)  # same as your code (typically N=20)

    # diagonal points: i=1..N-1, j=N-i  -> lr1=i*step_val, lr2=j*step_val
    xs = np.array([i * step_val for i in range(1, N)], dtype=float)
    lr2s = np.array([(N - i) * step_val for i in range(1, N)], dtype=float)

    steps_list = sorted(set(int(s) for s in steps_list))
    if any(s <= 0 for s in steps_list):
        raise ValueError("All steps in steps_list must be positive integers.")

    sum_losses = {s: np.zeros_like(xs, dtype=float) for s in steps_list}

    for seed in seeds:
        torch.manual_seed(seed)

        rho = 0.0
        #rho = 0.01

        # test input (Gaussian)
        x = torch.randn(test_size, d, device=device)

        # W1/W2 orthogonal init; X orthogonal columns scaled
        W1_init = orthonormal_columns(d, h, device=device, dtype=torch.float32)
        W2_init = orthonormal_columns(h, h, device=device, dtype=torch.float32)

        Qx = orthonormal_columns(n, d, device=device, dtype=torch.float32)
        X = torch.sqrt(torch.tensor(float(n), device=device)) * Qx

        xi = torch.normal(mean=0, std=np.sqrt(rho), size=(n, 1), device=device)               # n x 1
        a = torch.normal(mean=0, std=1/np.sqrt(h), size=(h, 1), device=device)                # h x 1
        beta = torch.normal(mean=0, std=1/np.sqrt(h), size=(d, 1), device=device)             # d x 1
        y = X @ beta + xi                                                                     # n x 1

        with torch.no_grad():
            xb = x @ beta  # (test_size x 1)

        for k in range(xs.shape[0]):
            lr1 = float(xs[k])
            lr2 = float(lr2s[k])

            W1 = W1_init.clone()
            W2 = W2_init.clone()

            # Run GD; record losses at each requested step
            t = 0
            for target_step in steps_list:
                while t < target_step:
                    W1, W2, _ = train_step(W1, W2, X, y, a, h, lr1, lr2)
                    t += 1

                with torch.no_grad():
                    inv_sqrt_h = 1.0 / torch.sqrt(torch.tensor(h, dtype=W1.dtype, device=W1.device))
                    pred_test = inv_sqrt_h * x @ W1 @ W2 @ a
                    resid = pred_test - xb
                    test_loss = torch.norm(resid, p='fro') ** 2 / x.size(0)

                sum_losses[target_step][k] += float(test_loss)

    avg_losses = {s: sum_losses[s] / len(seeds) for s in steps_list}
    return xs, avg_losses


def main():
    args = parse_args()

    # keep your behavior: force GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    d = int(args.d)
    h = int(args.h)
    n = int(args.n)
    test_size = int(args.test_size)

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

    # parse steps list
    steps_list = [int(s.strip()) for s in args.steps_list.split(",") if s.strip() != ""]
    if len(steps_list) == 0:
        raise ValueError("--steps-list is empty.")
    if any(s <= 0 for s in steps_list):
        raise ValueError("All steps in --steps-list must be positive integers.")
    steps_list = sorted(set(steps_list))

    os.makedirs(args.outdir, exist_ok=True)

    # DO NOT change your sizes
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

    # create a figure per requested step
    fig_step = {}
    ax_step = {}
    for s in steps_list:
        fig_step[s], ax_step[s] = plt.subplots(figsize=(8, 6))

    for lr_max in lr_max_values:
        xs, losses_by_step = compute_curves_for_one_lrmax(
            lr_max,
            d=d, h=h, n=n, test_size=test_size,
            device=device,
            seeds=seeds,
            steps_list=steps_list
        )

       
        step_val = max(1, lr_max // 20) if lr_max > 0 else 1
        N = int(lr_max / step_val)
        const = N * step_val
        x_mid = 0.5 * const
        idx_mid = int(np.argmin(np.abs(xs - x_mid)))

        color = cmap(norm(lr_max))
        for s in steps_list:
            ax_step[s].plot(xs, losses_by_step[s], linestyle='-', linewidth=4, color=color)

            
            ax_step[s].plot(
                xs[idx_mid], losses_by_step[s][idx_mid],
                marker='*', markersize=15,
                markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.8,
                linestyle='None', zorder=10
            )

    # style + colorbar + save
    for s in steps_list:
        ax = ax_step[s]
        fig = fig_step[s]

        ax.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.set_ylabel(r"Test Loss", fontsize=40)
        ax.set_xlabel(r" $\eta_1$", fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r" $\eta_1+\eta_2$")

        
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=20)

        out_pdf = os.path.join(
            args.outdir,
            f"3-NN-diagonal-multicurve_noise_{0.0}_step_{s}_lr_n_{n}_d_{d}_h_{h}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}_plot.pdf"
        )
        out_png = os.path.join(
            args.outdir,
            f"3-NN-diagonal-multicurve_noise_{0.0}_step_{s}_lr_n_{n}_d_{d}_h_{h}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}_plot.png"
        )

        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
