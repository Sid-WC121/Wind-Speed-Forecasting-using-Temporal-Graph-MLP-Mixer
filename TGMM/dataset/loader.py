import os
import torch
import xarray as xr
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tsl.ops.connectivity import adj_to_edge_index
from TGMM.dataset import get_datamodule
from TGMM.model.utils import StaticGraphTopologyData
from TGMM.model.transform import GraphPartitionTransform


def load_data(train_cfg):
    """
    Loads the wind dataset and prepares dataloaders and topology data.

    Args:
        train_cfg: The main training configuration (contains metis, pos_enc, train params)

    Returns:
        train_loader, val_loader, topo_data, metadata
    """
    # Load dataset config
    dataset_config_path = os.path.join(
        os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(
            f"Dataset config not found at {dataset_config_path}")

    data_cfg = OmegaConf.load(dataset_config_path)

    # Extract parameters
    ds_cfg = data_cfg.dataset

    # Load data
    features_path = os.path.join(ds_cfg.data_path, "features.nc")
    targets_path = os.path.join(ds_cfg.data_path, "targets.nc")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Targets file not found at {targets_path}")

    ds = xr.open_dataset(features_path, decode_timedelta=True)
    ds_targets = xr.open_dataset(targets_path, decode_timedelta=True)

    dm = get_datamodule(
        ds=ds,
        ds_targets=ds_targets,
        val_split=ds_cfg.val_split,
        test_start_date=ds_cfg.test_start_date,
        train_val_end_date=ds_cfg.train_val_end_date,
        lead_time_hours=ds_cfg.hours_leadtime,
        predictors=ds_cfg.predictors,
        target_var=ds_cfg.target_var,
        return_graph=True,
        graph_kwargs=ds_cfg.graph_kwargs,
    )

    # Create topology
    adj_matrix = dm.adj_matrix
    edge_index, edge_weight = adj_to_edge_index(adj=torch.tensor(adj_matrix))
    n_nodes = dm.train_dataset.stations
    topo_data = StaticGraphTopologyData(edge_index, edge_weight, n_nodes)

    # Apply METIS
    if train_cfg.metis.enable and train_cfg.metis.n_patches > 0:
        transform_train = GraphPartitionTransform(
            n_patches=train_cfg.metis.n_patches,
            metis=train_cfg.metis.enable,
            drop_rate=0,
            num_hops=train_cfg.metis.num_hops,
            is_directed=False,
            patch_rw_dim=train_cfg.pos_enc.patch_rw_dim,
            patch_num_diff=train_cfg.pos_enc.patch_num_diff
        )
        topo_data = transform_train(topo_data)

    metadata = dm.metadata

    # Create DataLoaders
    train_loader = DataLoader(
        dm.train_dataset, batch_size=train_cfg.train.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        dm.val_dataset, batch_size=train_cfg.train.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        dm.test_dataset, batch_size=train_cfg.train.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, topo_data, metadata
