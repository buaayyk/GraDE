import os
import sys

import torch

import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops


def to_dense(data, to_place_holder=False):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch) # (batch, n_node, n_node_type)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index,
                    batch=data.batch,
                    edge_attr=edge_attr,
                    max_num_nodes=max_num_nodes) # (batch, n_node, n_node, n_edge_type)
    E = encode_no_edge(E)
    if getattr(data, 'level', None) is not None:
        level, _ = to_dense_batch(x=data.level, batch=data.batch)
    else:
        level = None

    # random permutation
    bs, n_max = X.shape[0], X.shape[1]
    perm_indices = torch.argsort(torch.rand(bs, n_max, device=X.device), dim=1)
    X = X.gather(1, perm_indices.unsqueeze(-1).expand(bs, n_max, X.shape[-1]))
    E = E.gather(1, perm_indices.unsqueeze(-1).unsqueeze(-1).expand(bs, n_max, n_max, E.shape[3]))
    E = E.gather(2, perm_indices.unsqueeze(1).unsqueeze(-1).expand(bs, n_max, n_max, E.shape[3]))
    node_mask = node_mask.gather(1, perm_indices)
    if level is not None:
        level = level.gather(1, perm_indices)

    if to_place_holder:
        return PlaceHolder(X=X, E=E, y=None, level=level), node_mask
    return X, E, node_mask, level


def encode_no_edge(E):
    assert len(E.shape) == 4
    # if E.shape[-1] == 0:
    #     return E
    # TODO: check
    no_edge = torch.sum(E, dim=3) == 0
    E[:, :, :, 0][no_edge] = 1
    return E

def add_mask(X, E, node_mask):
    """
    Zero masking nonexisted node and edge features
    """
    x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
    e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
    e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

    X = X * x_mask
    E = E * e_mask1 * e_mask2
    # assert torch.allclose(E, torch.transpose(E, 1, 2))
    return X, E

def add_mask_idx(X_idx, E_idx, n_node_type, n_edge_type, node_mask):
    """
    Args:
    X_idx: (batch, n_node)
    E_idx: (batch, n_node, n_node)
    node_mask: (batch, n_node)

    For discrete data whose original category index is in [0, n_xxx_type)
    The masked idx is the "n_xxx_type"
    Note here the node_mask=True means the node should NOT be masked
    """
    X_mask = ~node_mask
    E_mask = ~(node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)) # as long as there is a node nonexisting, the edge should be masked out.
    # E_mask: (batch, n_node, n_node)
    batch_size, n_node = E_mask.shape[0], E_mask.shape[1]
    diagonal_mask = torch.eye(n_node, device=E_mask.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
    E_mask = E_mask | diagonal_mask

    if X_idx != None:
        X_idx_masked = torch.masked_fill(X_idx, X_mask, value=n_node_type)
    else:
        X_idx_masked = None
    E_idx_masked = torch.masked_fill(E_idx, E_mask, value=n_edge_type)

    # assert torch.allclose(E, torch.transpose(E, 1, 2))
    return X_idx_masked, E_idx_masked

def density(E, node_mask):
    """
    Args:
    E: (batch, n_node, n_node, n_node_type)
    node_mask: (batch, n_node)
    """
    pred_unconnnected = E[:,:,:,0] # (batch, n_node, n_node)
    E_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
    total_unconnected = (pred_unconnnected*E_mask).flatten().sum()
    total_node_pair = E_mask.flatten().sum()
    density = 1 - total_unconnected/total_node_pair

    return density

def density_idx(E, node_mask):
    """
    Args:
    E: (batch, n_node, n_node)
    node_mask: (batch, n_node)
    """
    E_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
    pred_unconnnected = (E == 0) * E_mask
    total_unconnected = pred_unconnnected.flatten().sum()
    total_node_pair = E_mask.flatten().sum()
    density = 1 - total_unconnected/total_node_pair

    return density

def batch_symmetrize(tensor):
    """
    Args:
    tensor: (batch, n, n)
    """
    assert len(tensor.shape) == 3
    assert tensor.shape[1] == tensor.shape[2]
    n = tensor.shape[1]
    u_mask = (torch.ones(n,n).triu() == 1)
    tensor.transpose(1,2)[:, u_mask] = tensor[:,u_mask]
    return tensor

class PlaceHolder:
    """
    From digress, used to mask nodes and edges
    """
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def to_graph_list(X, E, n_node):
    graph_list = []
    for i in range(X.shape[0]):
        n = n_node[i]
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()
        graph_list.append([atom_types, edge_types])
    return graph_list