from grade.utils.random_seed import RANDOM_SEED

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

from abc import ABC, abstractmethod
from grade.datasets.subgraph_sampling import sampling
from grade.datasets.subgraph_sampling.sampling import build_ground_truth, build_ground_truth_by_ESU, NRS, ARS, Rand_ESU, build_ground_truth_by_ESU_mp, Rand_ESU_mp, Rand_FaSE

class AbstractDataset(ABC, torch.utils.data.Dataset):

    def __init__(self, 
        root: str,
        split: Literal['train', 'valid', 'test', 'estimate'], 
        subgraph_sampling_method: str | list[str] = None, 
        subgraph_size: int | list [int] = None,
        sampling_density: float | list[float] = None,
        raw_file_name: str = "raw.pt",
        encoder: dict[str, int] = {},
        decoder: list[str] = [],
        directed_flag: bool = True,
        x_one_hot = True,
        tau: float = 0
    ):
        self.root = pathlib.Path(root)
        self.split = split
        
        self.subgraph_sampling_method = subgraph_sampling_method if isinstance(subgraph_sampling_method, list) else [subgraph_sampling_method] if subgraph_sampling_method is not None else None
        self.subgraph_size = subgraph_size if isinstance(subgraph_size, list) else [subgraph_size] if subgraph_size is not None else None
        self.sampling_density = sampling_density if isinstance(sampling_density, list) else [sampling_density] if sampling_density is not None else None
        if isinstance(self.sampling_density, list):
            self.sampling_density = [x if x <= 1 else int(x) for x in self.sampling_density]
        self.raw_data_path = self.root / raw_file_name
        self.preprocessed_data_path = self.root / "fullgraphs.pt"
        self.encoder = encoder
        self.decoder = decoder
        self.directed_flag = directed_flag
        self.x_one_hot = x_one_hot
        self.tau = tau
        
        if self.subgraph_sampling_method is None:
            assert self.subgraph_size is None and self.sampling_density is None
            self.topK_subgraphs_path = None
            self.topK_subgraphs_hash_path = None
            self.inter_data_path = self.preprocessed_data_path
            self.processed_data_dirpath = self.root / "Full"
        else:
            assert self.subgraph_size is not None and self.sampling_density is not None
            assert len(self.subgraph_sampling_method) == len(self.subgraph_size)
            assert len(self.subgraph_size) == len(self.sampling_density)
            self.topK_subgraphs_paths = [
                self.root / f"topK_subgraphs-{ss}.pt" for ss in self.subgraph_size
            ]
            self.topK_subgraphs_hash_paths = [
                self.root / f"topK_subgraphs_hash-{ss}.pt" for ss in self.subgraph_size
            ]
            subgraph_sampling_flag = "+".join([
                f"{self.subgraph_sampling_method[idx]}-{self.subgraph_size[idx]}-{self.sampling_density[idx]}" 
                for idx in range(len(self.subgraph_size))
            ])
            self.inter_data_path = self.root / f"subgraphs-{subgraph_sampling_flag}.pt"
            self.processed_data_dirpath = self.root / f"{subgraph_sampling_flag}"
        if split == "estimate":
            assert self.subgraph_size is not None
            # assert self.sampling_density[0] <= 1
            self.processed_data_path = self.topK_subgraphs_paths[0]
        else:
            self.processed_data_path = self.processed_data_dirpath / f"{split}.pt"
        self.dataset_info_path = self.processed_data_dirpath / "dataset_info.pt"
        
        self.processed_data = None
        self.dataset_info = None

    def prepare(self):
        assert self.raw_data_path.exists()

        if not self.preprocessed_data_path.exists():
            self.preprocess()
        
        if self.subgraph_sampling_method is not None and not self.inter_data_path.exists():
            self.subgraph_sampling()

        if not self.processed_data_path.exists() or not self.dataset_info_path.exists():
            self.process()

    def load_processed_data(self):
        self.prepare()
        print("loading [processed_data]")
        self.processed_data = torch.load(self.processed_data_path, weights_only=False)
        print("loading finish")

    def __len__(self):
        if self.processed_data is None:
            self.load_processed_data()
        return len(self.processed_data)

    def __getitem__(self, idx):
        if self.processed_data is None:
            self.load_processed_data()
        datapoint = self.processed_data[idx]
        if not self.x_one_hot:
            res = datapoint.clone()
            res.x = F.one_hot(res.x, num_classes=len(self.encoder)).float()
            return res
        else:
            return datapoint

    def get_dataset_info(self):
        if self.dataset_info is None:
            self.prepare()
            self.dataset_info = torch.load(self.dataset_info_path, weights_only=False)
        return self.dataset_info

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError

    def subgraph_sampling(self):
        print("[subgraph sampling]")    
        print(f"subgraph_sampling_method: {self.subgraph_sampling_method}")
        print(f"subgraph_size: {self.subgraph_size}")
        print(f"sampling_density: {self.sampling_density}")
        self.preprocessed_data = torch.load(self.preprocessed_data_path, weights_only=False)
        self.inter_data = []
        for idx in range(len(self.subgraph_size)):
            ssm = self.subgraph_sampling_method[idx]
            ss = self.subgraph_size[idx]
            sd = self.sampling_density[idx]
            assert sd >= 0
            non_isomorphism = True
            if sd <= 1:
                if self.topK_subgraphs_paths[idx].exists():
                    topK_subgraphs = torch.load(self.topK_subgraphs_paths[idx], weights_only=False)
                    topK_subgraphs_hash = torch.load(self.topK_subgraphs_hash_paths[idx], weights_only=False)
                else:
                    print("building subgraph ground truth")
                    if self.__class__.__name__ in {
                        "YoungerDataset", "YoungerPartDataset"
                    }:
                        topK_subgraphs, topK_subgraphs_hash = build_ground_truth_by_ESU_mp(self.preprocessed_data, ss, self.directed_flag)
                    elif getattr(self, "ground_truth_sampling_backend", None) == "ESU":
                        topK_subgraphs, topK_subgraphs_hash = build_ground_truth_by_ESU(self.preprocessed_data, ss, self.directed_flag)
                    else:
                        topK_subgraphs, topK_subgraphs_hash = build_ground_truth(self.preprocessed_data, ss, self.directed_flag)
                    
                    print(f"saving [topK_subgraphs] to {self.topK_subgraphs_paths[idx]}")
                    torch.save(topK_subgraphs, self.topK_subgraphs_paths[idx])
                    torch.save(topK_subgraphs_hash, self.topK_subgraphs_hash_paths[idx])
                    print("saving finish")
                non_isomorphism = True
                subgraph_dataset_size = int(len(topK_subgraphs) * sd)
            else:
                non_isomorphism = False
                subgraph_dataset_size = sd
            print("subgraph sampling")
            
            if ssm == "NRS":
                subgraphs = NRS(self.preprocessed_data, ss, subgraph_dataset_size, self.directed_flag, non_isomorphism)
            elif ssm == "ARS":
                subgraphs = ARS(self.preprocessed_data, ss, subgraph_dataset_size, self.directed_flag, non_isomorphism)
            elif ssm == "RESU":
                if self.__class__.__name__ in {
                    "YoungerDataset", "YoungerPartDataset"
                }:
                    use_N_max = True
                else:
                    use_N_max = False
                if sd >= 1:
                    subgraphs = Rand_ESU_mp(self.preprocessed_data, ss, subgraph_dataset_size, self.tau, self.directed_flag, non_isomorphism, 8, use_N_max)
                else:
                    subgraphs = Rand_ESU(self.preprocessed_data, ss, subgraph_dataset_size, self.tau, self.directed_flag, non_isomorphism, use_N_max)
            elif ssm == "RFASE":
                if self.__class__.__name__ in {
                    "YoungerDataset", "YoungerPartDataset"
                }:
                    use_N_max = True
                else:
                    use_N_max = False
                if sd >= 1:
                    if ss == 15:
                        parallel_splits = 2
                    else:
                        parallel_splits = 8
                    subgraphs = Rand_FaSE(self.preprocessed_data, ss, subgraph_dataset_size, self.directed_flag, non_isomorphism, self.tau, RANDOM_SEED, use_N_max, parallel_splits, None, False) # mp
                else:
                    print(non_isomorphism)
                    subgraphs = Rand_FaSE(self.preprocessed_data, ss, subgraph_dataset_size, self.directed_flag,  non_isomorphism, self.tau, RANDOM_SEED, use_N_max, 1, None, False)
            else:
                raise NotImplementedError
            self.inter_data.extend(subgraphs)
        print(f"saving [inter_data] to {self.inter_data_path}")
        torch.save(self.inter_data, self.inter_data_path)
        print(f"saving finish")

    def process(self):
        print("[process]")
        self.inter_data = torch.load(self.inter_data_path, weights_only=False)

        # Split dataset
        print("Spliting data")
        train_ratio = 0.98
        valid_ratio = 0.01
        test_ratio = 0.01

        train_size = int(len(self.inter_data) * train_ratio)
        valid_size = int(len(self.inter_data) * valid_ratio)
        test_size = len(self.inter_data) - train_size - valid_size
        if valid_size == 0:
            valid_size += 1
            train_size -= 1
        if test_size == 0:
            test_size += 1
            train_size -= 1

        torch.manual_seed(42)
        indices = torch.randperm(len(self.inter_data)).long().cpu().tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size+valid_size]
        test_indices = indices[train_size+valid_size:]

        train_data = [self.inter_data[i] for i in train_indices]
        valid_data = [self.inter_data[i] for i in valid_indices]
        test_data = [self.inter_data[i] for i in test_indices]

        self.processed_data_dirpath.mkdir(parents=True, exist_ok=True)

        print(f"saving [processed_data] to {self.processed_data_dirpath}")
        torch.save(train_data, self.processed_data_dirpath / "train.pt")
        torch.save(valid_data, self.processed_data_dirpath / "valid.pt")
        torch.save(test_data, self.processed_data_dirpath / "test.pt")
        print("saving finish")

        print(f"train / valid / test: {train_size} / {valid_size} / {test_size}")

        # Compute Dataset Information
        print("Computing Dataset Information")
        n_node_distribution = self.node_counts(train_data + valid_data).cpu().tolist()
        X_marginal = self.node_types(train_data).cpu().tolist()
        E_marginal = self.edge_types(train_data).cpu().tolist()
        max_n_nodes = len(n_node_distribution) - 1
        dataset_info = {
            "max_n_nodes": max_n_nodes,
            "n_node_type": len(X_marginal),
            "n_edge_type": len(E_marginal),
            "n_node_distribution": n_node_distribution,
            "X_marginal": X_marginal,
            "E_marginal": E_marginal, 
            "encoder": self.encoder,
            "decoder": self.decoder
        }
        print(f"saving [dataset_info] to {self.dataset_info_path}")
        torch.save(dataset_info, self.dataset_info_path)

    def node_counts(self, data):
        max_nodes = 0
        for i in tqdm.tqdm(range(len(data))):
            max_nodes = max(max_nodes, data[i].x.shape[0])

        all_counts = torch.zeros(max_nodes + 1)
        for i in tqdm.tqdm(range(len(data))):
            n_node = data[i].x.shape[0]
            all_counts[n_node] += 1
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self, data):
        n_node_type = len(self.encoder)
        distr = torch.zeros(n_node_type).float() # (n_node_type, ) distribution

        for i in tqdm.tqdm(range(len(data))):
            if self.x_one_hot:
                distr += data[i].x.sum(dim=0)
            else:
                distr += torch.bincount(data[i].x.to(torch.long), minlength=n_node_type).float()
        distr = distr / distr.sum()
        return distr

    def edge_types(self, data):
        n_edge_type = data[0].edge_attr.shape[1]
        distr = torch.zeros(n_edge_type).float() # (n_edge_type, ) distribution

        for i in tqdm.tqdm(range(len(data))):
            n_node = data[i].x.shape[0]
            n_edge = data[i].edge_index.shape[1]
            n_non_edge = n_node * (n_node - 1) - n_edge

            distr[1:] += data[i].edge_attr.sum(dim=0)[1:]
            distr[0] += n_non_edge
        
        distr = distr / distr.sum()
        return distr

    def get_fullgraph_dataset(self):
        if self.subgraph_sampling_method is None:
            return self
        else:
            return type(self)(
                root=self.root, 
                split=self.split,
                subgraph_sampling_method=None,
                subgraph_size=None,
                sampling_density=None
            )
