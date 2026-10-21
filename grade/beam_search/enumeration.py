import networkx
from tqdm import tqdm
import multiprocessing as mp
import torch

from grade.utils.format_transform import tensor_to_networkx

def _merge_shared(shared_dict, local_dict, lock):
    with lock:
        for h, info in local_dict.items():
            if h in shared_dict:
                entry = dict(shared_dict[h])
                entry['count'] += info['count']
                shared_dict[h] = entry
            else:
                shared_dict[h] = local_dict[h]

def _worker_2(score_dataset, left, left_lock, shared_dict, dict_lock, progress_q):
    while True:
        with left_lock:
            graph_idx = left.value
            left.value += 1
        if graph_idx >= len(score_dataset):
            break
        datapoint = score_dataset[graph_idx]
        G = tensor_to_networkx(datapoint, True)
        local = {}
        for u in G.nodes():
            for v in G.successors(u):
                sub_G = G.subgraph([u, v])
                hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
                if hash_sub_G not in local:
                    local[hash_sub_G] = {'count': 0, 'graph_idx': graph_idx, 'S': [u, v]}
                local[hash_sub_G]['count'] += 1
        _merge_shared(shared_dict, local, dict_lock)
        progress_q.put(1)

def _worker_3(score_dataset, left, left_lock, shared_dict, dict_lock, progress_q):
    while True:
        with left_lock:
            graph_idx = left.value
            left.value += 1
        if graph_idx >= len(score_dataset):
            break
        datapoint = score_dataset[graph_idx]
        G = tensor_to_networkx(datapoint, True)
        for node in G.nodes():
            G.nodes[node]['operator'] = torch.tensor(G.nodes[node]['operator'])
        for (u, v), attr in zip(
            datapoint.edge_index.t().cpu().tolist(),
            datapoint.edge_attr.cpu().tolist(),
        ):
            if G.has_edge(u, v):
                G[u][v]['edge_type'] = attr
        local = {}
        for u in G.nodes():
            neighbors = list(set(G.predecessors(u)) | set(G.successors(u)))
            for w_idx in range(len(neighbors)):
                w = neighbors[w_idx]
                for v_idx in range(w_idx + 1, len(neighbors)):
                    v = neighbors[v_idx]
                    if G.has_edge(w, v) or G.has_edge(v, w):
                        if u < w and u < v:
                            flag = True
                        else:
                            flag = False
                    else:
                        flag = True
                    if flag:
                        sub_G = G.subgraph([u, w, v])
                        hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
                        if hash_sub_G not in local:
                            local[hash_sub_G] = {'count': 0, 'graph_idx': graph_idx, 'S': [u, w, v]}
                        local[hash_sub_G]['count'] += 1
        _merge_shared(shared_dict, local, dict_lock)
        progress_q.put(1)

def enumerate_2_node_subgraph_mp(score_dataset, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    manager = mp.Manager()
    left = manager.Value('i', 0)
    left_lock = manager.Lock()
    dict_lock = manager.Lock()
    shared = manager.dict()
    progress_q = manager.Queue()

    procs = []
    for _ in range(num_workers):
        p = mp.Process(target=_worker_2, args=(score_dataset, left, left_lock, shared, dict_lock, progress_q))
        p.start()
        procs.append(p)

    total = len(score_dataset)
    with tqdm(total=total, desc="2-node") as pbar:
        processed = 0
        while processed < total:
            progress_q.get()
            pbar.update(1)
            processed += 1

    for p in procs:
        p.join()
    return dict(shared)

def enumerate_3_node_subgraph_mp(score_dataset, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    manager = mp.Manager()
    left = manager.Value('i', 0)
    left_lock = manager.Lock()
    dict_lock = manager.Lock()
    shared = manager.dict()
    progress_q = manager.Queue()

    procs = []
    for _ in range(num_workers):
        p = mp.Process(target=_worker_3, args=(score_dataset, left, left_lock, shared, dict_lock, progress_q))
        p.start()
        procs.append(p)

    total = len(score_dataset)
    with tqdm(total=total, desc="3-node") as pbar:
        processed = 0
        while processed < total:
            progress_q.get()
            pbar.update(1)
            processed += 1

    for p in procs:
        p.join()
    return dict(shared)

def enumerate_2_node_subgraph(score_dataset):
    hash_to_info = {}
    for graph_idx in tqdm(range(len(score_dataset))):
        datapoint = score_dataset[graph_idx]
        G = tensor_to_networkx(datapoint, True)
        for u in G.nodes():
            for v in G.successors(u):
                sub_G = G.subgraph([u, v])        
                hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
                if hash_sub_G not in hash_to_info.keys():
                    hash_to_info[hash_sub_G] = {
                        'count': 0,
                        'graph_idx': graph_idx,
                        'S': [u, v]
                    }
                hash_to_info[hash_sub_G]['count'] += 1
    return hash_to_info

def enumerate_3_node_subgraph(score_dataset):
    hash_to_info = {}
    for graph_idx in tqdm(range(len(score_dataset))):
        datapoint = score_dataset[graph_idx]
        G = tensor_to_networkx(datapoint, True)
        for node in G.nodes():
            G.nodes[node]['operator'] = torch.tensor(G.nodes[node]['operator'])
        for (u, v), attr in zip(
            datapoint.edge_index.t().cpu().tolist(),
            datapoint.edge_attr.cpu().tolist(),
        ):
            if G.has_edge(u, v):
                G[u][v]['edge_type'] = attr
        for u in G.nodes():
            neighbors = list(set(G.predecessors(u)) | set(G.successors(u)))
            for w_idx in range(len(neighbors)):
                w = neighbors[w_idx]
                for v_idx in range(w_idx + 1, len(neighbors)):
                    v = neighbors[v_idx]
                    if G.has_edge(w, v) or G.has_edge(v, w):
                        if u < w and u < v:
                            flag = True
                        else:
                            flag = False
                    else:
                        flag = True
                    if flag:
                        sub_G = G.subgraph([u, w, v])
                        hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
                        if hash_sub_G not in hash_to_info.keys():
                            hash_to_info[hash_sub_G] = {
                                'count': 0,
                                'graph_idx': graph_idx,
                                'S': [u, w, v]
                            }
                        hash_to_info[hash_sub_G]['count'] += 1
    return hash_to_info

import os
import pathlib
import json
from grade.datasets.loader.Younger import YoungerDataset

def enumerate_subgraphs(subgraph_size=None, num_workers=8):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root = pathlib.Path(cur_dir).parents[1] / "data" / "Younger"
    train_set = YoungerDataset(root, "train")
    val_set = YoungerDataset(root, "valid")
    test_set = YoungerDataset(root, "test")
    score_dataset = train_set + val_set + test_set

    if subgraph_size in (None, 2):
        hash_to_info = enumerate_2_node_subgraph_mp(score_dataset, num_workers=num_workers)
        with open(root / "all_2.json", "w") as f:
            json.dump(hash_to_info, f, indent=2)
    if subgraph_size in (None, 3):
        hash_to_info = enumerate_3_node_subgraph_mp(score_dataset, num_workers=num_workers)
        with open(root / "all_3.json", "w") as f:
            json.dump(hash_to_info, f, indent=2)

if __name__ == "__main__":
    enumerate_subgraphs()
