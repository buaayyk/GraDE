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
from torch_geometric.utils import to_undirected

from grade.datasets.loader.AbstractDataset import AbstractDataset
from grade.datasets.utils import topological_sorted_node_indices_from_edge_index, alloc_node_level_from_edge_index
import json

class YoungerPartDataset(AbstractDataset):

    def __init__(self, 
        root: str, 
        split: Literal['train', 'valid', 'test', 'estimate'], 
        subgraph_sampling_method: str = None, 
        subgraph_size: int = None,
        sampling_density: float = None
    ):
        # Customized setup for Younger
        raw_file_name: str = "Younger-Full-inf.zip"
        encoder_json_path = pathlib.Path(root) / "codebook.json"
        with open(encoder_json_path, "r") as f:
            encoder = json.load(f)
        decoder = [0 for _ in range(len(encoder))]
        for k, v in encoder.items():
            decoder[v] = k
        directed_flag = True
        x_one_hot = False
        tau = 2
        super().__init__(
            root, 
            split, 
            subgraph_sampling_method, 
            subgraph_size, 
            sampling_density, 
            raw_file_name,
            encoder,
            decoder,
            directed_flag,
            x_one_hot,
            tau
        )
    
    @overrides
    def preprocess(self):
        print("[preprocess]")
        self.raw_data = torch.load(self.raw_data_path, weights_only=False)
        temp = []
        for idx in range(0, len(self.raw_data), 10):
            temp.append(self.raw_data[idx])
        self.raw_data = temp
        print(f"len(self.raw_data)={len(self.raw_data)}")

        self.preprocessed_data = []
        
        for datapoint in tqdm.tqdm(self.raw_data):
            edge_index = np.array(datapoint['edge_index'])
            ops = np.array(datapoint['X'])
            n_nodes = len(ops)
            # TODO: topo sort or not
            topo_seq = topological_sorted_node_indices_from_edge_index(edge_index, n_nodes)

            ops = [ops[node_idx] for node_idx in topo_seq]
            if edge_index.shape[0] != 0 and edge_index.shape[1] != 0:
                inv = np.empty(n_nodes, dtype=np.int64)
                inv[topo_seq] = np.arange(n_nodes, dtype=np.int64)
                edge_index = np.vstack([
                    inv[edge_index[0]],
                    inv[edge_index[1]]
                ])
            level = alloc_node_level_from_edge_index(edge_index, n_nodes)
            # TODO: undirected graph or directed graph ?
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = to_undirected(edge_index, num_nodes=n_nodes)

            ops = torch.tensor(ops)
            x = ops
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
    root = pathlib.Path(cur_dir) / ".." / ".." / ".." / "data" / "YoungerPart"
    dataset = YoungerPartDataset(root, "train", "NRS", 5, 0.1)
    print(dataset.get_dataset_info())
    dataset[0]