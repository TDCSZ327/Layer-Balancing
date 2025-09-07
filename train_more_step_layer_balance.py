import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_steps = 16 ##number of steps
test_size = 500
seeds = range(2020, 2025)

lr_max=9001 ## max_lr

lr1_range = np.arange(0, lr_max, lr_max//20)
lr2_range = np.arange(0, lr_max, lr_max//20)

plt.rcParams.update(
    {"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 20, "legend.fontsize": 14}
)

loss_grid = np.zeros((len(lr1_range), len(lr2_range)))

def train_step(W1, W2, X, y, a, h, lr1, lr2):

    n = X.size(0)

   
    W1_ = W1.detach().clone().requires_grad_(True)
    W2_ = W2.detach().clone().requires_grad_(True)

    pred = (1.0 / h) * X @ W1_ @ W2_ @ a
    loss = (0.5 / n) * torch.norm(pred - y, p='fro')**2
    loss.backward()

    with torch.no_grad():
        W1_new = W1_ - float(lr1) * W1_.grad
        W2_new = W2_ - float(lr2) * W2_.grad

    return W1_new.detach(), W2_new.detach(), loss.detach()

for seed in seeds:
    torch.manual_seed(seed)

   
    d = 1000
    h = 1000
    n = 1000
    rho = 0.0


    x = torch.randn(test_size, d, device=device)
    W1_init = torch.normal(mean=0, std=1/np.sqrt(d), size=(d, h), device=device)   # d x h
    W2_init = torch.normal(mean=0, std=1/np.sqrt(h), size=(h, h), device=device)   # h x h
    X = torch.randn(n, d, device=device)                                           # n x d
    xi = torch.normal(mean=0, std=np.sqrt(rho), size=(n, h), device=device)        # n x h
    a = torch.eye(h, device=device)                                                # h x h
    beta = torch.normal(mean=0, std=1/d, size=(d, h), device=device)               # d x h
    y = X @ beta + xi                                                               # n x h

    for i, lr1 in enumerate(lr1_range):
        for j, lr2 in enumerate(lr2_range):
            W1 = W1_init.clone()
            W2 = W2_init.clone()

            last_train_loss = None
            for _ in range(int(N_steps)):
                W1, W2, last_train_loss = train_step(W1, W2, X, y, a, h, lr1, lr2)

            with torch.no_grad():
                pred_test = (1.0 / h) * x @ W1 @ W2 @ a
                resid = pred_test - (x @ beta)
                test_loss = torch.norm(resid, p='fro')**2 / x.size(0)

            
            loss_grid[i, j] += float(test_loss)

loss_grid /= len(seeds)

min_val = np.min(loss_grid)
min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
i, j = min_idx
print(f"min Loss: {min_val}")
print(f"minima i={i}, j={j}")
print(f"optimal lr1={lr1_range[i]}, lr2={lr2_range[j]}")

plt.figure(figsize=(8, 6))
plt.imshow(
    loss_grid,
    aspect='auto',
    cmap='viridis',
    origin='lower',
    extent=[lr2_range[0], lr2_range[-1], lr1_range[0], lr1_range[-1]],
)
plt.colorbar(label='Loss')
plt.xlabel('lr2')
plt.ylabel('lr1')
plt.title('2-layer-NN Loss', fontsize=20)
plt.tight_layout()
plt.savefig(f'step_{N_steps}_lr_{lr_max}.png')
print(f'step_{N_steps}_lr_{lr_max}.png')
