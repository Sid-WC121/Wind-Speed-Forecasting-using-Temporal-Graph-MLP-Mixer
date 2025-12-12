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


def log_prediction_plots(x, y, pred_dist, example_indices, stations, epoch, input_denormalizer, model_name="", plot_dir=".", seed=None):
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
    seed_str = f"_seed_{seed}" if seed is not None else ""
    plot_filename = os.path.join(
        plot_dir, f"{model_name}_predictions_epoch_{epoch}{seed_str}.png")
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
    seed=None,
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
    seed_str = f"_seed_{seed}" if seed is not None else ""
    outpath = os.path.join(plot_dir, f"rankhist_all{seed_str}.png")
    fig.savefig(outpath)
    print(f"Saved rank histograms to {outpath}")
    plt.close(fig)


def plot_pooled_rank_histogram(
    model,
    dataloader: DataLoader,
    edge_index,
    model_name: str = "",
    n_samples: int = 20,
    plot_dir: str = ".",
    seed=None,
):
    """
    Compute and plot a pooled rank histogram (Talagrand diagram) 
    over ALL lead times and ALL stations.
    """
    device = next(model.parameters()).device
    model.eval()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.to(device)

    # Use incremental histogram to avoid memory issues
    hist = torch.zeros(n_samples + 1, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, y, valid_x, valid_y = batch
                valid_x = valid_x.to(device)
            else:
                x, y = batch
                valid_x = None

            x = x.to(device)                                  # [B, L_in, N, F]
            y = y.to(device).squeeze(-1)                      # [B, L_out, N]
            y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)

            if hasattr(model, "forward") and model_name != "baseline":
                if valid_x is not None:
                    dist = model(x, valid_x)
                else:
                    dist = model(x, edge_index)
            else:
                dist = model(x)

            samp = dist.rsample((n_samples,)).squeeze(-1) 
            
            y_flat = y.view(-1)                          # [M]
            samp_flat = samp.reshape(n_samples, -1)      # [n_samples, M]

            mask = ~torch.isnan(y_flat)
            y_flat = y_flat[mask]                        # [M_valid]
            samp_flat = samp_flat[:, mask]               # [n_samples, M_valid]

            if y_flat.shape[0] == 0:
                continue
            
            ranks = (samp_flat < y_flat.unsqueeze(0)).sum(dim=0) 
            
            # Accumulate histogram incrementally (avoids storing all ranks)
            batch_hist = torch.bincount(ranks.cpu(), minlength=n_samples + 1)
            hist += batch_hist

    total_count = hist.sum().item()
    if total_count == 0:
        print("No valid targets found for pooled rank histogram.")
        return

    plot_dir = os.path.join(plot_dir, model_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Normalize to probability density
    hist_normalized = hist.float() / total_count

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(n_samples + 1), hist_normalized.numpy(), width=0.8, edgecolor="k")
    ax.set_title(f"Pooled Rank Histogram (All Leads) — {model_name}")
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Rank")
    ax.set_xlim(-0.5, n_samples+0.5)
    
    # Plot ideal flat line
    ax.axhline(1.0/(n_samples+1), color='r', linestyle='--', label='Ideal')
    ax.legend()

    plt.tight_layout()
    seed_str = f"_seed_{seed}" if seed is not None else ""
    outpath = os.path.join(plot_dir, f"rankhist_pooled{seed_str}.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved pooled rank histogram to {outpath}")


def plot_single_lead_rank_histogram(
    model,
    dataloader: DataLoader,
    edge_index,
    lead_hour: int = 1,
    model_name: str = "",
    n_samples: int = 20,
    plot_dir: str = ".",
    seed=None,
):
    """
    Compute and plot a rank histogram for a SINGLE lead time with ideal line.
    
    Args:
        lead_hour: The specific lead hour to plot (e.g., 1 for 1h forecast)
    """
    device = next(model.parameters()).device
    model.eval()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.to(device)

    # Use incremental histogram to avoid memory issues
    hist = torch.zeros(n_samples + 1, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, y, valid_x, valid_y = batch
                valid_x = valid_x.to(device)
            else:
                x, y = batch
                valid_x = None

            x = x.to(device)
            y = y.to(device).squeeze(-1)  # [B, L, S]
            y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)

            if hasattr(model, "forward") and model_name != "baseline":
                if valid_x is not None:
                    dist = model(x, valid_x)
                else:
                    dist = model(x, edge_index)
            else:
                dist = model(x)

            samp = dist.rsample((n_samples,)).squeeze(-1)  # [n_samples, B, L, S]
            
            # Extract the specific lead time
            if lead_hour < y.shape[1]:
                y_h = y[:, lead_hour, :]  # [B, S]
                samp_h = samp[:, :, lead_hour, :]  # [n_samples, B, S]
                
                y_flat = y_h.reshape(-1)
                samp_flat = samp_h.reshape(n_samples, -1)
                
                mask = ~torch.isnan(y_flat)
                y_flat = y_flat[mask]
                samp_flat = samp_flat[:, mask]
                
                if y_flat.shape[0] == 0:
                    continue
                
                ranks = (samp_flat < y_flat.unsqueeze(0)).sum(dim=0)
                batch_hist = torch.bincount(ranks.cpu(), minlength=n_samples + 1)
                hist += batch_hist

    total_count = hist.sum().item()
    if total_count == 0:
        print(f"No valid targets found for lead {lead_hour}h rank histogram.")
        return

    plot_dir = os.path.join(plot_dir, model_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Normalize to relative frequency
    hist_np = hist.float().numpy() / total_count
    ranks = np.arange(n_samples + 1)  # 0..20

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(ranks, hist_np, width=0.8, edgecolor='black')
    ax.set_title(f"Rank histogram - lead t={lead_hour}h", fontsize=14)
    ax.set_ylabel("Relative frequency", fontsize=12)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_xticks(ranks)
    ax.grid(axis='y', alpha=0.3)
    
    # Expected uniform line
    expected_freq = 1.0 / len(hist_np)
    ax.axhline(y=expected_freq, color='red', linestyle='--', linewidth=2, label=f'Expected (uniform): {expected_freq:.4f}')
    ax.legend()

    plt.tight_layout()
    seed_str = f"_seed_{seed}" if seed is not None else ""
    outpath = os.path.join(plot_dir, f"rankhist_lead_{lead_hour}h{seed_str}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {lead_hour}h lead rank histogram to {outpath}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f'Model loaded from {checkpoint_path}, starting at epoch {epoch}')
    return model, optimizer, epoch


def plot_loss_curves(train_losses, val_losses, train_maes, val_maes, model_name="", plot_dir=".", seed=None):
    """
    Plot training and validation loss curves (CRPS and MAE).
    
    Args:
        train_losses: List of training CRPS values per epoch
        val_losses: List of validation CRPS values per epoch
        train_maes: List of training MAE values per epoch
        val_maes: List of validation MAE values per epoch
        model_name: Name of the model for plot title
        plot_dir: Directory to save the plot
        seed: Random seed used for training
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
    seed_str = f"_seed_{seed}" if seed is not None else ""
    plot_filename = os.path.join(plot_dir, f"{model_name}_loss_curves{seed_str}.png")
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
