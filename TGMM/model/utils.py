import torch


class StaticGraphTopologyData(object):
    """
    Static graph topology data fed into METIS subgraph sampler.
    Required by GMMModel for graph partitioning.

    Usage:
        from tsl.ops.connectivity import adj_to_edge_index
        edge_index, edge_weight = adj_to_edge_index(adj_matrix)
        topo_data = StaticGraphTopologyData(edge_index, edge_weight, n_nodes)
        transform = GraphPartitionTransform(n_patches=80, metis=True, ...)
        topo_data = transform(topo_data)
    """

    def __init__(self, edge_index, edge_weight, num_nodes):
        self.edge_index = torch.tensor(edge_index) if not isinstance(
            edge_index, torch.Tensor) else edge_index
        self.edge_weight = torch.tensor(edge_weight) if not isinstance(
            edge_weight, torch.Tensor) else edge_weight
        self.num_nodes = num_nodes

        assert self.edge_weight.shape[0] == self.edge_index.shape[
            1], "Edge weight must be a 1D tensor with the same length as the number of edges"
        assert self.edge_weight.ndim == 1, "Edge weight must be a 1D tensor"

    def __repr__(self):
        return f"StaticGraphTopologyData(edge_index={self.edge_index.shape}, edge_weight={self.edge_weight.shape}, num_nodes={self.num_nodes})"

    def __str__(self):
        return self.__repr__()
