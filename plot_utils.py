import numpy as np
import matplotlib.pyplot as plt

def plot_progress_grid(hist, save_name, figsize=(60, 20)):
    """
    hist: array of shape (2, N, L)
          hist[0] = train loss per epoch per position
          hist[1] = val loss per epoch per position
    """

    train_hist = hist[:, 0]   # (N, L)
    val_hist   = hist[:, 1]   # (N, L)
    N = train_hist.shape[0]
    L = train_hist.shape[1]

    # Grid size
    n_rows = 3
    n_cols = int(np.ceil(L / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=False)
    axes = axes.flatten()

    for i in range(L):
        ax = axes[i]
        ax.plot(train_hist[:, i], label="train", color="blue", alpha=0.7)
        ax.plot(val_hist[:, i], label="val", color="red", alpha=0.7)
        ax.set_title(f"Position {i}")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(L, len(axes)):
        axes[j].axis("off")

    axes[0].legend()
    fig.suptitle("Training / Validation Loss per Position", fontsize=16)

    plt.savefig(f"plots/{save_name}.png", dpi=150)
    plt.close()
