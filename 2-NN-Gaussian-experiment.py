#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "mathtext.fontset": "stix",         
    "pdf.fonttype": 42,                
    "ps.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser(
        description="2-NN (Gaussian init): sweep lr_max and plot diagonal curves for arbitrary steps-list with lr_max colorbar"
    )

    # keep your original args
    p.add_argument("--d", type=int, default=1000, help="Input dimension (d)")
    p.add_argument("--h", type=int, default=1000, help="Hidden width (h)")
    p.add_argument("--n", type=int, default=1000, help="Train set size (n)")
    p.add_argument("--test-size", type=int, default=500, help="Test set size")

    # NEW: lr_max sweep (you want 30..40 step 1)
    p.add_argument("--lr-max-start", type=int, default=27000, help="Start lr_max (inclusive)")
    p.add_argument("--lr-max-end", type=int, default=63000, help="End lr_max (inclusive)")
    p.add_argument("--lr-max-step", type=int, default=2000, help="Step for lr_max sweep")

    # NEW: arbitrary steps to record
    p.add_argument("--steps-list", type=str, default="1,2,4,8",
                   help="Comma-separated steps to record, e.g. '1,2,4,8'")

    # output dir (match your style)
    p.add_argument("--outdir", type=str, default="_", help="Output directory")

    return p.parse_args()


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


def compute_diagonal_curves_for_one_lrmax(
    lr_max, *,
    N_steps_max,
    d, h, n, test_size,
    device, seeds,
    steps_list
):
    """
    For a given lr_max:
      - use step_val = max(1, lr_max//20) exactly like your code
      - lr ranges: np.arange(0, lr_max, step_val)
      - N = int(lr_max/step_val)
      - diagonal indices: (i, j) with i=1..N-1, j=N-i (same as your code)
      - compute test loss at each requested step in steps_list.

    Returns:
      xs: np.array of eta1 values along diagonal
      losses_by_step: dict {step: np.array avg test loss along diagonal}
    """
    step_val = max(1, lr_max // 20) if lr_max > 0 else 1
    N = int(lr_max / step_val)

    if N <= 1:
        raise ValueError(f"lr_max={lr_max} too small for diagonal (N={N}).")

    xs = np.array([i * step_val for i in range(1, N)], dtype=float)                 # eta1
    lr2s = np.array([(N - i) * step_val for i in range(1, N)], dtype=float)         # eta2 (index-diagonal)

    steps_list = sorted(set(int(s) for s in steps_list))
    if any(s <= 0 for s in steps_list):
        raise ValueError("All steps in --steps-list must be positive integers.")
    if max(steps_list) > N_steps_max:
        raise ValueError("Internal error: max(steps_list) > N_steps_max")

    sum_losses = {s: np.zeros_like(xs, dtype=float) for s in steps_list}

    for seed in seeds:
        torch.manual_seed(seed)

        rho = 0.0

        # SAME as your code (Gaussian init)
        x = torch.randn(test_size, d, device=device)
        W1_init = torch.normal(mean=0, std=1/np.sqrt(d), size=(d, h), device=device)   # d x h
        W2_init = torch.normal(mean=0, std=1/np.sqrt(h), size=(h, h), device=device)   # h x h
        X = torch.randn(n, d, device=device)                                           # n x d
        xi = torch.normal(mean=0, std=np.sqrt(rho), size=(n, h), device=device)        # n x h
        a = torch.eye(h, device=device)                                                # h x h
        beta = torch.normal(mean=0, std=1/d, size=(d, h), device=device)               # d x h
        y = X @ beta + xi                                                              # n x h

        with torch.no_grad():
            xb = x @ beta  # (test_size x h)

        # diagonal points
        for idx in range(xs.shape[0]):
            lr1 = float(xs[idx])
            lr2 = float(lr2s[idx])

            W1 = W1_init.clone()
            W2 = W2_init.clone()

            # run GD up to N_steps_max and record at requested steps
            for t in range(1, N_steps_max + 1):
                W1, W2, _ = train_step(W1, W2, X, y, a, h, lr1, lr2)

                if t in sum_losses:
                    with torch.no_grad():
                        pred_test = (1.0 / h) * x @ W1 @ W2 @ a
                        resid = pred_test - xb
                        test_loss = torch.norm(resid, p='fro')**2 / x.size(0)

                    sum_losses[t][idx] += float(test_loss)

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

    lr_max_start = int(args.lr_max_start)
    lr_max_end = int(args.lr_max_end)
    lr_max_step = int(args.lr_max_step)

    if lr_max_step <= 0:
        raise ValueError("--lr-max-step must be positive.")
    if lr_max_end < lr_max_start:
        raise ValueError("--lr-max-end must be >= --lr-max-start.")

    lr_max_values = list(range(lr_max_start, lr_max_end + 1, lr_max_step))
    if len(lr_max_values) == 0:
        raise ValueError("No lr_max values to plot; check start/end/step.")

    steps_list = [int(s.strip()) for s in args.steps_list.split(",") if s.strip() != ""]
    if len(steps_list) == 0:
        raise ValueError("--steps-list is empty.")
    if any(s <= 0 for s in steps_list):
        raise ValueError("All steps in --steps-list must be positive integers.")
    steps_list = sorted(set(steps_list))
    max_steps_needed = max(steps_list)

    os.makedirs(args.outdir, exist_ok=True)

    # DO NOT change your sizes
    plt.rcParams.update(
        {"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 20, "legend.fontsize": 14}
    )

    seeds = range(2020, 2025)

    # colorbar mapping for lr_max
    # cmap = plt.get_cmap("viridis")
    # norm = mpl.colors.Normalize(vmin=float(min(lr_max_values)), vmax=float(max(lr_max_values)))
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

    # create a figure per requested step
    fig_step = {}
    ax_step = {}
    for s in steps_list:
        fig_step[s], ax_step[s] = plt.subplots(figsize=(8, 6))

    # sweep lr_max, add curves to each step-figure
    for lr_max in lr_max_values:
        xs, losses_by_step = compute_diagonal_curves_for_one_lrmax(
            lr_max,
            N_steps_max=max_steps_needed,
            d=d, h=h, n=n, test_size=test_size,
            device=device, seeds=seeds,
            steps_list=steps_list
        )

        color = cmap(norm(float(lr_max)))
        for s in steps_list:
            ax_step[s].plot(xs, losses_by_step[s], linestyle='-', linewidth=4, color=color)

    # style + colorbar + save
    for s in steps_list:
        ax = ax_step[s]
        fig = fig_step[s]

        ax.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.set_ylabel(r"Test Loss", fontsize=40)
        ax.set_xlabel(r" $\eta_1$", fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\eta_1+\eta_2$")

        out_pdf = os.path.join(
            args.outdir,
            f"2-NN-diagonal-multicurve_step_{s}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}_n_{n}_d_{d}_h_{h}.pdf"
        )
        out_png = os.path.join(
            args.outdir,
            f"2-NN-diagonal-multicurve_step_{s}_lrmax_{lr_max_start}_{lr_max_end}_step_{lr_max_step}_n_{n}_d_{d}_h_{h}.png"
        )

        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
