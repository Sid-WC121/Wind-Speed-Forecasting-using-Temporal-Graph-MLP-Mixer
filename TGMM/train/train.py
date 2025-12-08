import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import random
import time
from datetime import timedelta
import optuna
from optuna.samplers import GridSampler
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from TGMM.utils import load_config, mask_anomalous_targets, log_prediction_plots, plot_rank_histogram, save_checkpoint, load_checkpoint, plot_loss_curves
from TGMM.model.model import GMMModel
from TGMM.model.optimizer import get_optimizer
from TGMM.dataset.loader import load_data
import TGMM.model.losses as loss_prob
from tsl.ops.connectivity import adj_to_edge_index


MASK_ANOM = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'OUTPUTS')
checkpoint_dir = os.path.join(OUTPUTS_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)


def grid_space():
    return {
        # "knn" : [5, 10, 15],
        # "theta" : ["median", "factormedian", "std"],
        # "seed" : [3,4,5],
        "lr": [0.001, 0.0005, 0.0001]
    }


def get_search_space(trial):
    space = {
        "hidden_channels": 32,
        "num_layers": 2,
        "dropout_p": 0.2,
        "lr": trial.suggest_categorical("lr", [0.001, 0.0005, 0.0001]),
        "weight_decay": 1e-5,
        "scheduler": "CosineAnnealingWarmRestarts",
        "knn": 5,
        "threshold": 0.6,
        "theta": "std",
        "seed": 0,
    }

    if space["scheduler"] == "onecycle":
        space["pct_start"] = 0.1
    elif space["scheduler"] == "CosineAnnealingWarmRestarts":
        space["T_0"] = 10 
        space["T_mult"] = 1  
    elif space["scheduler"] == "cosine":
        space["t_max"] = 100
    elif space["scheduler"] == "steplr":
        space["step_size"] = 1
        space["step_gamma"] = 0.5
    else:
        space["exp_gamma"] = 0.9564069612741963
    return space


def update_config_with_trial(cfg: DictConfig, search_space):

    cfg.model.nfeatures_node = search_space['hidden_channels']
    cfg.model.nfeatures_patch = search_space['hidden_channels'] * 2
    cfg.model.nlayer_gnn = search_space['num_layers']

    if not hasattr(cfg.model, 'kwargs'):
        cfg.model.kwargs = OmegaConf.create({})

    cfg.model.kwargs.hidden_channels = search_space['hidden_channels']
    cfg.model.kwargs.num_layers = search_space['num_layers']
    cfg.model.kwargs.dropout_p = search_space['dropout_p']

    cfg.train.lr = search_space['lr']
    cfg.train.wd = search_space['weight_decay']

    if not hasattr(cfg, 'graph_kwargs'):
        cfg.graph_kwargs = OmegaConf.create({})
    cfg.graph_kwargs.knn = search_space['knn']
    cfg.graph_kwargs.threshold = search_space['threshold']
    cfg.graph_kwargs.theta = search_space['theta']

    cfg.seed = search_space["seed"]

    if not hasattr(cfg.train, 'scheduler'):
        cfg.train.scheduler = OmegaConf.create(
            {'algo': search_space['scheduler'], 'kwargs': {}})

    cfg.train.scheduler.algo = search_space['scheduler']

    if 'pct_start' in search_space:
        cfg.train.scheduler.kwargs.pct_start = search_space['pct_start']
    if 'T_0' in search_space:
        cfg.train.scheduler.kwargs.T_0 = search_space['T_0']
    if 'T_mult' in search_space:
        cfg.train.scheduler.kwargs.T_mult = search_space['T_mult']
    if 't_max' in search_space:
        cfg.train.scheduler.kwargs.t_max = search_space['t_max']
    if 'step_size' in search_space:
        cfg.train.scheduler.kwargs.step_size = search_space['step_size']
        cfg.train.scheduler.kwargs.step_gamma = search_space['step_gamma']
        cfg.train.scheduler.kwargs.gamma = search_space['step_gamma']
    if 'exp_gamma' in search_space:
        cfg.train.scheduler.kwargs.exp_gamma = search_space['exp_gamma']

    return cfg


def train_trial(cfg: DictConfig):
    print("Config:", cfg)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(cfg.seed if cfg.seed else 0)

    train_dataloader, val_dataloader, test_dataloader, topo_data, metadata = load_data(cfg)

    model = GMMModel(cfg, topo_data, metadata)

    epochs = cfg.train.epochs

    criterion = loss_prob.MaskedCRPSLogNormal()
    mae_crit = loss_prob.MaskedMAE()

    optimizer_name = cfg.train.get('optimizer', 'AdamW')
    optimizer_kwargs = {}
    if hasattr(cfg.train, 'ns_iters'):
        optimizer_kwargs['ns_iters'] = cfg.train.ns_iters
    if hasattr(cfg.train, 'momentum'):
        optimizer_kwargs['momentum'] = cfg.train.momentum
    if hasattr(cfg.train, 'grad_clip_norm'):
        optimizer_kwargs['grad_clip_norm'] = cfg.train.grad_clip_norm
    if hasattr(cfg.train, 'track_stats'):
        optimizer_kwargs['track_stats'] = cfg.train.track_stats
    
    optimizer = get_optimizer(
        optimizer_name,
        model.named_parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.wd,
        **optimizer_kwargs
    )

    USE_PRETRAINED = False
    if USE_PRETRAINED:
        _, _, start_epoch = load_checkpoint(model, optimizer, USE_PRETRAINED)

    scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.train.scheduler.algo)
    import inspect
    valid_params = inspect.signature(scheduler_cls.__init__).parameters
    filtered_kwargs = {
        k: v for k, v in cfg.train.scheduler.kwargs.items() if k in valid_params}
    lr_scheduler = scheduler_cls(optimizer, **filtered_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for attr_name in dir(topo_data):
        if not attr_name.startswith('_'):
            attr = getattr(topo_data, attr_name)
            if isinstance(attr, torch.Tensor):
                if attr.is_floating_point():
                    setattr(topo_data, attr_name, attr.float().to(device))
                else:
                    setattr(topo_data, attr_name, attr.to(device))
    
    edge_index = topo_data.edge_index

    print(f"Topology data attributes: {[attr for attr in dir(topo_data) if not attr.startswith('_')]}")
    required_attrs = ['subgraphs_nodes_mapper', 'combined_subgraphs', 'subgraphs_batch', 'subgraphs_edges_mapper']
    for attr in required_attrs:
        if hasattr(topo_data, attr):
            val = getattr(topo_data, attr)
            device_str = str(val.device) if hasattr(val, 'device') else 'N/A'
            print(f"  {attr}: {type(val)} shape={val.shape if hasattr(val, 'shape') else 'N/A'} device={device_str}")
        else:
            print(f"  {attr}: MISSING!")

    print("device", device)
    print('Training started.')
    print(f"Starting {epochs} epochs of training...")

    lowest_val_loss = float("inf")
    train_crps_history = []
    val_crps_history = []
    train_mae_history = []
    val_mae_history = []
    train_start_time = time.time()

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        model.train()
        train_crps_sum = 0.0
        train_mae_sum = 0.0

        for i, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch} Train", leave=False, position=1):
            if len(batch) == 4:
                x, y, valid_x, valid_y = batch
                valid_x = valid_x.to(device)
            else:
                x, y = batch
                valid_x = None

            x = x.to(device)
            if MASK_ANOM:
                y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)
            y = y.to(device)

            optimizer.zero_grad()

            if valid_x is not None:
                dist = model(x, valid_x, debug=(i==0 and epoch==0))
            else:
                dist = model(x, edge_index, debug=(i==0 and epoch==0))

            loss = criterion(dist, y)

            mu = dist.mean.squeeze(-1)
            truth = y.squeeze(-1)
            mae, _ = mae_crit(mu, truth)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step(epoch + i/len(train_dataloader))

            train_crps_sum += loss.item()
            train_mae_sum += mae.item()

        avg_train_crps = train_crps_sum / len(train_dataloader)
        avg_train_mae = train_mae_sum / len(train_dataloader)
        
        train_crps_history.append(avg_train_crps)
        train_mae_history.append(avg_train_mae)

        # ——— VALIDATION ——————————————————————————————————————
        model.eval()
        val_crps_sum = 0.0
        val_mae_sum = 0.0

        horizons = [1, 24, 48, 96]
        sum_crps = {h: 0.0 for h in horizons}
        count_crps = {h: 0 for h in horizons}
        sum_mae = {h: 0.0 for h in horizons}
        count_mae = {h: 0 for h in horizons}

        with torch.no_grad():
            firstbatch = True
            plot_rank_histogram(model, val_dataloader, edge_index, model_name=cfg.model.type if hasattr(
                cfg.model, 'type') else "GMMModel", plot_dir=checkpoint_dir)

            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} Valid", leave=False, position=1):
                if len(batch) == 4:
                    x, y, valid_x, valid_y = batch
                    valid_x = valid_x.to(device)
                else:
                    x, y = batch
                    valid_x = None

                x = x.to(device)
                if MASK_ANOM:
                    y = mask_anomalous_targets(
                        y, min_speed=0.2, max_speed=10.0)
                y = y.to(device)

                if valid_x is not None:
                    dist = model(x, valid_x)
                else:
                    dist = model(x, edge_index)

                loss = criterion(dist, y)

                mu = dist.mean.squeeze(-1)
                truth = y.squeeze(-1)
                mae, _ = mae_crit(mu, truth)

                val_crps_sum += loss.item()
                val_mae_sum += mae.item()

                for h in horizons:
                    if h < mu.shape[1]:
                        mu_h = mu[:, h, :]
                        y_h = truth[:, h, :]
                        valid = (~torch.isnan(y_h)).sum().item()

                        loc_h = dist.loc[:, h, :]
                        scale_h = dist.scale[:, h, :]
                        d_h = torch.distributions.LogNormal(loc_h, scale_h)
                        c_h = criterion(d_h, y_h).item()

                        sum_crps[h] += c_h * valid
                        count_crps[h] += valid

                        m_h, _ = mae_crit(mu_h, y_h)
                        sum_mae[h] += m_h.item() * valid
                        count_mae[h] += valid

                if firstbatch:
                    firstbatch = False
                    B_size = x.shape[0]
                    example_indices = [0, 0, 1, 1]
                    stations = [1, 2, 3, 4]
                    example_indices = [i % B_size for i in example_indices]
                    stations = [s % x.shape[2] for s in stations]

                    input_mean = torch.tensor(metadata['input_mean'])
                    input_std = torch.tensor(metadata['input_std'])

                    def input_denormalizer(x):
                        dev = x.device
                        m = input_mean.to(dev)
                        s = input_std.to(dev)
                        return x * s + m

                    log_prediction_plots(x=x, y=y, pred_dist=dist,
                                        example_indices=example_indices,
                                        stations=stations,
                                        epoch=epoch,
                                        input_denormalizer=input_denormalizer,
                                        model_name=cfg.model.type if hasattr(
                                            cfg.model, 'type') else "GMMModel",
                                        plot_dir=os.path.join(OUTPUTS_DIR, "plots"))

        avg_val_crps = val_crps_sum / len(val_dataloader)
        avg_val_mae = val_mae_sum / len(val_dataloader)
        avg_crps_h = {
            h: (sum_crps[h]/count_crps[h] if count_crps[h] > 0 else 0) for h in horizons}
        avg_mae_h = {h: (sum_mae[h]/count_mae[h]
                        if count_mae[h] > 0 else 0) for h in horizons}
        
        val_crps_history.append(avg_val_crps)
        val_mae_history.append(avg_val_mae)

        print(avg_val_crps)
        print(avg_val_mae)
        print(avg_crps_h)
        print(avg_mae_h)

        if avg_val_crps < lowest_val_loss:
            lowest_val_loss = avg_val_crps

        print(f"\n=== Validation Metrics @ Epoch {epoch+1} ===")
        print(f"Train CRPS ", avg_train_crps)
        print(f"Train MAE ", avg_train_mae)
        print(f"Validation CRPS ", avg_val_crps)
        print(f"Validation MAE ", avg_val_mae)
        print(f"{'Lead(h)':>8} │ {'CRPS':>8} │ {'MAE':>8}")
        print("-" * 30)
        for h in [1, 24, 48, 96]:
            print(f"{h:8d} │ {avg_crps_h[h]:8.4f} │ {avg_mae_h[h]:8.4f}")

        save_checkpoint(epoch, model, optimizer, checkpoint_dir,
                        name=cfg.model.type if hasattr(cfg.model, 'type') else "GMMModel")

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time
    train_time_str = str(timedelta(seconds=int(train_duration)))
    print(f"\n{'='*50}")
    print(f"Training completed in {train_time_str} (HH:MM:SS)")
    print(f"{'='*50}\n")

    plot_loss_curves(
        train_losses=train_crps_history,
        val_losses=val_crps_history,
        train_maes=train_mae_history,
        val_maes=val_mae_history,
        model_name=cfg.model.type if hasattr(cfg.model, 'type') else "GMMModel",
        plot_dir=checkpoint_dir
    )

    evaluate(cfg, model, test_dataloader, edge_index, device)  

    return lowest_val_loss

# ——— TEST ——————————————————————————————————————


def evaluate(cfg: DictConfig, model, test_dataloader, edge_index, device):
    model.eval()
    test_start_time = time.time()
    print(f"\n{'='*50}")
    print("Testing started...")
    print(f"{'='*50}\n")

    def test_model(model, loader, device):
        means, stds = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    x_batch, y_batch, valid_x, valid_y = batch
                    valid_x = valid_x.to(device)
                else:
                    x_batch, y_batch = batch
                    valid_x = None

                x_batch = x_batch.to(device)

                if valid_x is not None:
                    dist = model(x_batch, valid_x)
                else:
                    dist = model(x_batch, edge_index)

                means.append(dist.loc.cpu().numpy())
                stds.append(dist.scale.cpu().numpy())
        return np.concatenate(means, axis=0), np.concatenate(stds, axis=0)

    mean_vals, std_vals = test_model(model, test_dataloader, device)

    test_outputs_dir = os.path.join(OUTPUTS_DIR, 'test_outputs')
    os.makedirs(test_outputs_dir, exist_ok=True)
    output_file = os.path.join(
        test_outputs_dir, f'test_outputs_{cfg.model.type if hasattr(cfg.model, "type") else "GMMModel"}_seed_{cfg.seed}.npz')
    np.savez_compressed(output_file, mean=mean_vals, std=std_vals)
    print(f"Saved test outputs to {output_file}")

    mae_crit = loss_prob.MaskedMAE()
    criterion = loss_prob.MaskedCRPSLogNormal()

    with torch.no_grad():
        test_mae_sum = 0.0
        test_crps_sum = 0.0

        horizons = [1, 24, 48, 96]
        sum_crps = {h: 0.0 for h in horizons}
        count_crps = {h: 0 for h in horizons}
        sum_mae = {h: 0.0 for h in horizons}
        count_mae = {h: 0 for h in horizons}

        for i, batch in enumerate(test_dataloader):
            print(f"testing batch {i} of {len(test_dataloader)}")
            if len(batch) == 4:
                x_batch, y_batch, valid_x, valid_y = batch
                valid_x = valid_x.to(device)
            else:
                x_batch, y_batch = batch
                valid_x = None

            x = x_batch.to(device)
            y = y_batch.to(device)

            if valid_x is not None:
                dist = model(x, valid_x)
            else:
                dist = model(x, edge_index)

            mu = dist.mean.squeeze(-1)
            truth = y.squeeze(-1)
            loss_crps = criterion(dist, y)
            mae, _ = mae_crit(mu, truth)
            test_mae_sum += mae.item()
            test_crps_sum += loss_crps.item()

            for h in horizons:
                if h < mu.shape[1]:
                    mu_h = mu[:, h, :]
                    y_h = truth[:, h, :]
                    valid = (~torch.isnan(y_h)).sum().item()

                    loc_h = dist.loc[:, h, :]
                    scale_h = dist.scale[:, h, :]  
                    d_h = torch.distributions.LogNormal(loc_h, scale_h)
                    c_h = criterion(d_h, y_h).item()

                    sum_crps[h] += c_h * valid
                    count_crps[h] += valid

                    m_h, _ = mae_crit(mu_h, y_h)
                    sum_mae[h] += m_h.item() * valid
                    count_mae[h] += valid

            if i == 0:
                pass

        avg_crps_h = {
            h: (sum_crps[h]/count_crps[h] if count_crps[h] > 0 else 0) for h in horizons}
        avg_mae_h = {h: (sum_mae[h]/count_mae[h]
                        if count_mae[h] > 0 else 0) for h in horizons}
        
        avg_test_crps = test_crps_sum / len(test_dataloader)
        avg_test_mae = test_mae_sum / len(test_dataloader)
        
        print(f"\n=== Test Metrics ===")
        print(f"Test CRPS  {avg_test_crps}")
        print(f"Test MAE   {avg_test_mae}")
        print(f" Lead(h) │     CRPS │      MAE")
        print(f"{'─'*30}")
        for h in [1, 24, 48, 96]:
            print(f"{h:8d} │ {avg_crps_h[h]:8.4f} │ {avg_mae_h[h]:8.4f}")

        plot_rank_histogram(model, test_dataloader, edge_index, model_name=cfg.model.type if hasattr(
            cfg.model, 'type') else "GMMModel", plot_dir=os.path.join(OUTPUTS_DIR, "rank_histograms"))
        
        print("\nGenerating test prediction plots...")
        for i, batch in enumerate(test_dataloader):
            if i > 0:
                break
            if len(batch) == 4:
                x_batch, y_batch, valid_x, valid_y = batch
                valid_x = valid_x.to(device)
            else:
                x_batch, y_batch = batch
                valid_x = None

            x = x_batch.to(device)
            y = y_batch.to(device)

            if valid_x is not None:
                dist = model(x, valid_x)
            else:
                dist = model(x, edge_index)
            
            example_indices = [0, 0, 0, 0]
            stations = [0, 1, 50, 100]
            
            log_prediction_plots(
                x=x_batch,
                y=y_batch,
                pred_dist=dist,
                example_indices=example_indices,
                stations=stations,
                epoch="test",
                input_denormalizer=lambda x: x,
                model_name=cfg.model.type if hasattr(cfg.model, 'type') else "GMMModel",
                plot_dir=os.path.join(OUTPUTS_DIR, 'checkpoints', cfg.model.type if hasattr(cfg.model, 'type') else "GMMModel")
            )

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        test_time_str = str(timedelta(seconds=int(test_duration)))
        print(f"\n{'='*50}")
        print(f"Testing completed in {test_time_str} (HH:MM:SS)")
        print(f"{'='*50}\n")

        return avg_test_crps


def set_trial(trial):
    configs_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'configs')
    cfg = load_config(configs_dir=configs_dir)

    dataset_config_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'dataset', 'config.yaml')
    if os.path.exists(dataset_config_path):
        dataset_cfg = OmegaConf.load(dataset_config_path)
        cfg = OmegaConf.merge(cfg, dataset_cfg)
        if not hasattr(cfg.dataset, 'horizon'):
            cfg.dataset.horizon = cfg.dataset.hours_leadtime + 1

    search_space = get_search_space(trial)
    cfg = update_config_with_trial(cfg, search_space)

    return train_trial(cfg)


if __name__ == '__main__':
    study_name = "TGMM_Optimization_v2" 
    study = optuna.create_study(
        study_name=study_name, direction="minimize", sampler=GridSampler(grid_space()), load_if_exists=False)

    study.optimize(set_trial, n_trials=1)

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.params}")
