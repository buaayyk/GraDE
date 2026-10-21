from grade.datasets.utils import graph_hash
from grade.datasets.loader.AbstractDataset import AbstractDataset

import torch
from grade.datasets.utils import graph_hash

def get_hash2count(dataset: AbstractDataset):
    hash2count = dict()
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        X = torch.argmax(datapoint.x, dim=-1).cpu().tolist()
        edge_index = datapoint.edge_index.cpu().tolist()
        edge_attr = torch.argmax(datapoint.edge_attr, dim=-1).cpu().tolist()
        if getattr(datapoint, "level", None) is None:
            level = None
        else:
            level = datapoint.level.cpu().tolist()
        hash = graph_hash(X, edge_index, edge_attr, level is not None, level)
        hash2count[hash] = hash2count.get(hash, 0) + 1
    return hash2count

def get_estimate_values(train_set: AbstractDataset, estimate_set: AbstractDataset):
    hash2count = get_hash2count(train_set)
    estimate_values = []
    for idx in range(len(estimate_set)):
        datapoint = estimate_set[idx]
        X = torch.argmax(datapoint.x, dim=-1).cpu().tolist()
        edge_index = datapoint.edge_index.cpu().tolist()
        edge_attr = torch.argmax(datapoint.edge_attr, dim=-1).cpu().tolist()
        if getattr(datapoint, "level", None) is None:
            level = None
        else:
            level = datapoint.level.cpu().tolist()
        hash = graph_hash(X, edge_index, edge_attr, level is not None, level)
        estimate_values.append(hash2count.get(hash, 0))
    return estimate_values