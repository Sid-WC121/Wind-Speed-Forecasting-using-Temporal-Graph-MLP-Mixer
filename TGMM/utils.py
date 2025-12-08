import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def load_config(configs_dir: str):
    config_path = os.path.join(configs_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    return cfg


def mask_anomalous_targets(y, min_speed, max_speed):
    squeezed = (y.squeeze(-1) if y.dim() == 4 else y)
    bad = (squeezed < min_speed) | (
        squeezed > max_speed) | torch.isnan(squeezed)
    y_clean = squeezed.clone()
    y_clean[bad] = float('nan')
    return y_clean.unsqueeze(-1) if y.dim() == 4 else y_clean


def log_prediction_plots(x, y, pred_dist, example_indices, stations, epoch, input_denormalizer, model_name="", plot_dir="."):
    x = input_denormalizer(x)  # bring inputs to their original range
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs = axs.flatten()

    '''
    quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).repeat(
        *y.shape).to(pred_dist.mean.device)
    quantiles = pred_dist.icdf(quantile_levels).detach().cpu().numpy()
    # quantiles = np.swapaxes(quantiles, 1, 2)
    # print(quantiles)
    '''
    # pred_dist shape: [B, T, N] (from readout)
    # We want quantiles shape: [B, T, N, 5]
    q_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                            device=pred_dist.mean.device).view(5, 1, 1, 1)
    quantiles = pred_dist.icdf(q_levels)  # [5, B, T, N]
    quantiles = quantiles.permute(1, 2, 3, 0).detach().cpu().numpy()  # [B, T, N, 5]

    time = np.arange(x.shape[1])


    for i, (b_idx, station) in enumerate(zip(example_indices, stations)):
        ax = axs[i]
        ax.plot(x[b_idx, :, station, 0], label='ens_mean', color='forestgreen')
        ax.fill_between(time, quantiles[b_idx, :, station, 0], quantiles[b_idx, :, station, 1],
                        alpha=0.15, color="blue", label="5%-95%")

        ax.fill_between(time, quantiles[b_idx, :, station, 1], quantiles[b_idx, :, station, 2],
                        alpha=0.35, color="blue", label="25%-75%")

        ax.plot(time, quantiles[b_idx, :, station, 2],
                color="black", linestyle="--", label="Median (50%)")

        ax.fill_between(time, quantiles[b_idx, :, station, 2], quantiles[b_idx, :, station, 3],
                        alpha=0.35, color="blue")

        ax.fill_between(time, quantiles[b_idx, :, station, 3], quantiles[b_idx, :, station, 4],
                        alpha=0.15, color="blue")

        ax.plot(y[b_idx, :, station, 0],
                label='observed', color='mediumvioletred')
        ax.set_title(f'Station {station} at batch element {b_idx}')
        ax.set_xlabel("Lead time")
        ax.set_ylabel("Wind speed")

    axs[-1].legend()  # only show legend in the last plot

    plt.suptitle(f'Predictions at Epoch {epoch} for model {model_name}')
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(
        plot_dir, f"{model_name}_predictions_epoch_{epoch}.png")
    plt.savefig(plot_filename)
    plt.close(fig)


def plot_rank_histogram(
    model,
    dataloader: DataLoader,
    edge_index,
    dm=None,
    model_name: str = "",
    n_samples: int = 20,
    horizons: list = [1, 24, 48, 96],
    plot_dir: str = ".",
):
    """
    model       -- your trained model (already .to(device) and with state_dict loaded)
    dataloader  -- e.g. DataLoader(dm.val_dataset, batch_size=32, shuffle=False)
    edge_index  -- from adj_to_edge_index(dm.adj_matrix)
    dm          -- your datamodule (for denormalizer if needed)
    model_name  -- one of "baseline","tcn_gnn","bidirectionalstgnn"
    n_samples   -- number of trajectories to sample (20)
    horizons    -- list of lead‐time indices to compute histograms for
    """
    device = next(model.parameters()).device
    model.eval()
    edge_index = edge_index.to(device)

    ranks = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, y, valid_x, valid_y = batch
                valid_x = valid_x.to(device)
            else:
                x, y = batch
                valid_x = None

            x = x.to(device)                                  # [B, L, N,  F]
            y = y.to(device).squeeze(-1)                      # [B, L, N]
            y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)

            if hasattr(model, "forward") and model_name != "baseline":
                if valid_x is not None:
                    dist = model(x, valid_x)
                else:
                    dist = model(x, edge_index)
            else:
                dist = model(x)

            samp = dist.rsample((n_samples,)).squeeze(-1).cpu().numpy()
            truth = y.cpu().numpy()

            for h in horizons:
                below = (samp[:, :, horizons.index(h), :] <
                        truth[:, horizons.index(h), :][None, :, :])
                s_h = samp[:, :, h, :]
                t_h = truth[:, h, :]
                r_h = np.sum(s_h < t_h[None, :, :], axis=0)
                ranks[h].extend(r_h.flatten().tolist())

    plot_dir = plot_dir + f"/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()
    for i, h in enumerate(horizons):
        axs[i].hist(ranks[h], bins=np.arange(n_samples+2)-0.5, edgecolor="k")
        axs[i].set_title(f"Rank histogram — lead {h}h")
        axs[i].set_ylabel("Frequency")
        axs[i].set_xlim(-0.5, n_samples+0.5)

    plt.tight_layout()
    outpath = os.path.join(plot_dir, "rankhist_all.png")
    fig.savefig(outpath)
    print(f"Saved rank histograms to {outpath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f'Model loaded from {checkpoint_path}, starting at epoch {epoch}')
    return model, optimizer, epoch


def plot_loss_curves(train_losses, val_losses, train_maes, val_maes, model_name="", plot_dir="."):
    """
    Plot training and validation loss curves (CRPS and MAE).
    
    Args:
        train_losses: List of training CRPS values per epoch
        val_losses: List of validation CRPS values per epoch
        train_maes: List of training MAE values per epoch
        val_maes: List of validation MAE values per epoch
        model_name: Name of the model for plot title
        plot_dir: Directory to save the plot
    """
    epochs = np.arange(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, marker='o', label='Train CRPS', color='blue')
    ax1.plot(epochs, val_losses, marker='s', label='Val CRPS', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CRPS Loss')
    ax1.set_title(f'CRPS Loss Curves - {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_maes, marker='o', label='Train MAE', color='blue')
    ax2.plot(epochs, val_maes, marker='s', label='Val MAE', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'MAE Curves - {model_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, f"{model_name}_loss_curves.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"Saved loss curves to {plot_filename}")


def save_checkpoint(epoch, model, optimizer, checkpoint_dir, name=""):
    checkpoint_path = os.path.join(
        checkpoint_dir, f'model_{name}_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f'Model saved to {checkpoint_path}')
