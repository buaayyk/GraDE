import os
import torch
import torch.nn.functional as F
import torch_geometric
import random
import tqdm
import networkx
import numpy as np
import pathlib
from typing import Union, Literal
from overrides import overrides

from grade.datasets.loader.AbstractDataset import AbstractDataset
from grade.datasets.utils import topological_sorted_node_indices, alloc_node_level

class NASBench201Dataset(AbstractDataset):

    def __init__(self, 
        root: str, 
        split: Literal['train', 'valid', 'test', 'estimate'], 
        subgraph_sampling_method: str = None, 
        subgraph_size: int = None,
        sampling_density: float = None
    ):
        # Customized setup for NAS-Bench-201
        raw_file_name: str = "cache_cifar100.zip"
        decoder = ['out', 'inp', '1x1', '3x3', 'ap', 'skip', 'none']
        encoder = {decoder[i]: i for i in range(len(decoder))}
        directed_flag = True
        super().__init__(
            root, 
            split, 
            subgraph_sampling_method, 
            subgraph_size, 
            sampling_density, 
            raw_file_name,
            encoder,
            decoder,
            directed_flag
        )

    @overrides
    def preprocess(self):
        print("[preprocess]")
        self.raw_data = torch.load(self.raw_data_path, weights_only=False)
        self.preprocessed_data = []

        for datapoint in tqdm.tqdm(self.raw_data):
            ops = datapoint['x'].numpy()
            edge_index = datapoint['edge_index'].numpy()
            adj = np.zeros((len(ops), len(ops)))
            for edge_idx in range(edge_index.shape[1]):
                u = edge_index[0][edge_idx]
                v = edge_index[1][edge_idx]
                adj[u][v] = 1
            topo_seq = topological_sorted_node_indices(adj)
            adj = adj[np.ix_(topo_seq, topo_seq)]
            ops = [ops[node_idx] for node_idx in topo_seq]
            level = alloc_node_level(adj)
            
            # TODO: undirected graph or directed graph ?
            adj = adj.T + adj
            
            ops = torch.tensor(ops)
            x = F.one_hot(ops, num_classes=len(self.encoder)).float()
            edge_index = torch.tensor(np.array(np.nonzero(adj))).long()
            # TODO: torch.zeros(edge_index.shape[-1], 2).float() or torch.zeros(edge_index.shape[-1], 1).float() ?
            edge_attr = torch.zeros(edge_index.shape[-1], 2).float()
            edge_attr[:, 1] = 1
            # TODO: conditional or unconditional ?
            y = torch.tensor([[0]]).float()
            # y = torch.tensor([val_acc]).float()
            level = torch.tensor(level).float()

            preprocessed_datapoint = torch_geometric.data.Data(x=x, edge_index= edge_index, edge_attr=edge_attr, y=y, level=level)
            self.preprocessed_data.append(preprocessed_datapoint)

        print(f"saving [preprocessed_data] to {self.preprocessed_data_path}")
        torch.save(self.preprocessed_data, self.preprocessed_data_path)
        print("saving finsh")

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root = pathlib.Path(cur_dir) / ".." / ".." / ".." / "data" / "NASBench201"
    train_set = NASBench201Dataset(root, "train", None, None, None)
    val_set = NASBench201Dataset(root, "valid", None, None, None)
    test_set = NASBench201Dataset(root, "test", None, None, None)
    all_set = train_set + val_set + test_set
    print(len(train_set))
    print(len(val_set))
    print(len(test_set))
    print(len(all_set))