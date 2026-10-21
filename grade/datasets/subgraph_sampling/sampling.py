from grade.utils.random_seed import RANDOM_SEED

import tqdm
import networkx
import random
import numpy as np
import torch_geometric
import torch
import os
import pathlib
import warnings
import json
import time
from itertools import combinations
from collections import deque, defaultdict
import multiprocessing as mp
import math

warnings.filterwarnings('ignore')

from grade.datasets.subgraph_sampling.single_graph_sampling import neighbour_reservoir_sampling

from grade.datasets.utils import graph_hash, realloc_subgraph_node_level
from grade.utils.format_transform import tensor_to_networkx, networkx_to_tensor
from grade.datasets.subgraph_sampling.GTRIE import construct_gtrie, extract_subgraph, GraphRepresentationCache
from grade.datasets.subgraph_sampling.FaSE import FaSEEngine, build_neighbour_cache

def build_ground_truth(data, subgraph_size, directed_flag=True):
    hash_to_info = {}
    # ground truth
    for graph_idx in tqdm.tqdm(range(len(data))):
        datapoint = data[graph_idx]
        n_node = datapoint.x.shape[0]
        lst = list(range(n_node))
        adj = np.zeros((n_node, n_node))
        adj_attr = np.zeros((n_node, n_node))
        edge_index = datapoint.edge_index.cpu().tolist()
        edge_attr = datapoint.edge_attr.cpu().tolist()
        if directed_flag:
            level = datapoint.level.cpu().tolist()
        for edge_idx in range(len(edge_index[0])):
            u = edge_index[0][edge_idx]
            v = edge_index[1][edge_idx]
            if not directed_flag or level[u] < level[v]:
                adj[u][v] = 1
                adj_attr[u][v] = np.argmax(np.array(edge_attr[edge_idx]))
        for sub_lst in combinations(lst, subgraph_size):
            sub_lst = list(sub_lst)
            bfs_flags = deque([sub_lst[0]])
            bfs_queues = deque([sub_lst[0]])

            while len(bfs_queues) > 0:
                size = len(bfs_queues)
                while size > 0:
                    u = bfs_queues.popleft()
                    for v in sub_lst:
                        if u != v and (adj[u][v] == 1 or adj[v][u] == 1):
                            if v not in bfs_flags:
                                bfs_flags.append(v)
                                bfs_queues.append(v)
                    size -= 1
            if len(bfs_flags) == len(sub_lst):
                G = networkx.DiGraph()
                sub_adj = adj[np.ix_(sub_lst, sub_lst)]
                sub_adj_attr = adj_attr[np.ix_(sub_lst, sub_lst)]
                sub_x_temp = datapoint.x[sub_lst]
                # sub_X = torch.argmax(datapoint.x[sub_lst], dim=1).cpu().tolist()
                sub_X = torch.argmax(sub_x_temp, dim=1).cpu().tolist() if sub_x_temp.dim() == 2 else sub_x_temp.cpu().tolist()
                for i in range(len(sub_X)):
                    G.add_node(i, operator = sub_X[i])
                for i in range(len(sub_X)):
                    for j in range(len(sub_X)):
                        if sub_adj[i][j] == 1:
                            G.add_edge(i, j, edge_type=sub_adj_attr[i][j])
                hash_G = networkx.weisfeiler_lehman_graph_hash(G, node_attr='operator', edge_attr='edge_type')
                if hash_G in hash_to_info.keys():
                    hash_to_info[hash_G].count += 1
                else:
                    sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                        subset=sub_lst,
                        edge_index=datapoint.edge_index,
                        edge_attr=datapoint.edge_attr,
                        relabel_nodes=True,
                        num_nodes=datapoint.x.shape[0]
                    )
                    sub_x = datapoint.x[sub_lst]
                    sub_y = datapoint.y
                    if directed_flag:
                        sub_level = realloc_subgraph_node_level(len(sub_lst), sub_edge_index, datapoint.level[sub_lst].numpy())
                        sub_level = torch.tensor(sub_level).float()
                        subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level, count=torch.tensor([1]).long().to(datapoint.x.device))
                    else:
                        subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, count=torch.tensor([1]).long().to(datapoint.x.device))
                    
                    hash_to_info[hash_G] = subgraph_datapoint
    
    sorted_items = sorted(hash_to_info.items(), key=lambda kv: kv[1].count.item(), reverse=True)
    topK_subgraphs = [info for h, info in sorted_items]
    topK_subgraphs_hash = [h for h, info in sorted_items]

    return topK_subgraphs, topK_subgraphs_hash

def build_ground_truth_by_ESU(data, subgraph_size, directed_flag=True):
    hash_to_info = {}
    for graph_idx in tqdm.tqdm(range(len(data))):
        datapoint = data[graph_idx]
        n_node = datapoint.x.shape[0]
        G = tensor_to_networkx(datapoint, directed_flag=directed_flag)
        neighbors = [set() for _ in range(n_node)]
        for (u, v) in G.edges():
            neighbors[u].add(v)
            neighbors[v].add(u)
        def handle_subgraph(V_subgraph):
            V_subgraph_list = list(V_subgraph)
            sub_G = G.subgraph(V_subgraph)
            hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
            if hash_sub_G in hash_to_info.keys():
                hash_to_info[hash_sub_G].count += 1
            else:
                sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                    subset=V_subgraph_list,
                    edge_index=datapoint.edge_index,
                    edge_attr=datapoint.edge_attr,
                    relabel_nodes=True,
                    num_nodes=datapoint.x.shape[0]
                )
                sub_x = datapoint.x[V_subgraph_list]
                sub_y = datapoint.y
                if directed_flag:
                    sub_level = realloc_subgraph_node_level(len(V_subgraph_list), sub_edge_index, datapoint.level[V_subgraph_list].numpy())
                    sub_level = torch.tensor(sub_level).float()
                    subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level, count=torch.tensor([1]).long().to(datapoint.x.device))
                else:
                    subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, count=torch.tensor([1]).long().to(datapoint.x.device))
                hash_to_info[hash_sub_G] = subgraph_datapoint
        
        def extend_subgraph(V_subgraph, V_extension, root):
            if len(V_subgraph) == subgraph_size:
                handle_subgraph(V_subgraph)
                return
            while len(V_extension) != 0:
                w = V_extension.pop()
                exclusion_set = set()
                for v in V_subgraph:
                    exclusion_set.update(neighbors[v])                
                new_extension = set([
                    u for u in neighbors[w]
                    if u > root 
                    and u not in exclusion_set
                ])
                V_extension_ = V_extension | new_extension
                extend_subgraph(V_subgraph | {w}, V_extension_, root)
        
        root_iter = range(n_node)
        if len(data) == 1:
            root_iter = tqdm.tqdm(
                root_iter,
                desc=f"ESU roots (k={subgraph_size})",
                leave=True,
            )
        for root in root_iter:
            V_extension = set([u for u in neighbors[root] if u > root])
            extend_subgraph({root}, V_extension, root)
            if len(data) == 1 and hasattr(root_iter, "set_postfix"):
                root_iter.set_postfix(
                    unique=len(hash_to_info),
                    refresh=False,
                )
    
    sorted_items = sorted(hash_to_info.items(), key=lambda kv: (-kv[1].count.item(), kv[0]))
    topK_subgraphs = [info for h, info in sorted_items]
    topK_subgraphs_hash = [h for h, info in sorted_items]

    return topK_subgraphs, topK_subgraphs_hash

def _merge_shared_build_ground_truth_by_ESU(shared_dict, local_dict, lock):
    with lock:
        for h, info in local_dict.items():
            if h in shared_dict:
                entry = dict(shared_dict[h])
                entry['count'] += info['count']
                shared_dict[h] = entry
            else:
                shared_dict[h] = info

def _worker_build_ground_truth_by_ESU(data, subgraph_size, directed_flag, left, left_lock, shared_dict, dict_lock, progress_q):
    while True:
        with left_lock:
            graph_idx = left.value
            left.value += 1
        if graph_idx >= len(data):
            break
        datapoint = data[graph_idx]
        n_node = datapoint.x.shape[0]
        G = tensor_to_networkx(datapoint, directed_flag=directed_flag)
        neighbors = [set() for _ in range(n_node)]
        for (u, v) in G.edges():
            neighbors[u].add(v)
            neighbors[v].add(u)
        local = {}
        local_count = 0
        flush_threshold = 10 ** 6
        def handle_subgraph(V_subgraph):
            nonlocal local_count
            V_subgraph_list = list(V_subgraph)
            sub_G = G.subgraph(V_subgraph)
            hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
            if hash_sub_G not in local:
                local[hash_sub_G] = {'count': 0, 'graph_idx': graph_idx, 'S': V_subgraph_list}
            local[hash_sub_G]['count'] += 1
            local_count += 1
            if local_count >= flush_threshold:
                _merge_shared_build_ground_truth_by_ESU(shared_dict, local, dict_lock)
                local.clear()
                local_count = 0
        def extend_subgraph(V_subgraph, V_extension, root):
            if len(V_subgraph) == subgraph_size:
                handle_subgraph(V_subgraph)
                return
            while len(V_extension) != 0:
                w = V_extension.pop()
                exclusion_set = set()
                for v in V_subgraph:
                    exclusion_set.update(neighbors[v])
                new_extension = set([
                    u for u in neighbors[w]
                    if u > root and u not in exclusion_set
                ])
                V_extension_ = V_extension | new_extension
                extend_subgraph(V_subgraph | {w}, V_extension_, root)
        for root in range(n_node):
            V_extension = set([u for u in neighbors[root] if u > root])
            extend_subgraph({root}, V_extension, root)
        _merge_shared_build_ground_truth_by_ESU(shared_dict, local, dict_lock)
        progress_q.put(graph_idx)

def build_ground_truth_by_ESU_mp(data, subgraph_size, directed_flag=True, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    manager = mp.Manager()
    left = manager.Value('i', 0)
    left_lock = manager.Lock()
    dict_lock = manager.Lock()
    shared = manager.dict()
    progress_q = manager.Queue()
    procs = []
    for _ in range(max(1, min(num_workers, len(data)))):
        p = mp.Process(target=_worker_build_ground_truth_by_ESU, args=(data, subgraph_size, directed_flag, left, left_lock, shared, dict_lock, progress_q))
        p.start()
        procs.append(p)
    total = len(data)
    processed_graph_idxs = set()
    end_t = time.time() + 24 * 3600
    with tqdm.tqdm(total=total, desc=f"ESU-{subgraph_size}-node(mp)") as pbar:
        processed = 0
        while processed < total:
            left_t = max(end_t - time.time(), 0)
            try:
                idx = progress_q.get(timeout=left_t)
                processed_graph_idxs.add(int(idx))
                pbar.update(1)
                processed += 1
            except Exception:
                alive = False
                for p in procs:
                    if p.is_alive():
                        alive = True
                        break
                if not alive or left_t <= 0:
                    break
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join()
    merged = dict(shared)
    
    sorted_items = sorted(merged.items(), key=lambda kv: (-kv[1]['count'], kv[0]))
    topK_subgraphs = []
    topK_subgraphs_hash = []
    for h, info in sorted_items:
        graph_idx = info['graph_idx']
        S = info['S']
        datapoint = data[graph_idx]
        sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
            subset=S,
            edge_index=datapoint.edge_index,
            edge_attr=datapoint.edge_attr,
            relabel_nodes=True,
            num_nodes=datapoint.x.shape[0]
        )
        sub_x = datapoint.x[S]
        sub_y = datapoint.y
        count_tensor = torch.tensor([info['count']]).long()
        if directed_flag:
            sub_level = realloc_subgraph_node_level(len(S), sub_edge_index, datapoint.level[S].numpy())
            sub_level = torch.tensor(sub_level).float()
            sub_data = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level, count=count_tensor)
        else:
            sub_data = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, count=count_tensor)
        topK_subgraphs.append(sub_data)
        topK_subgraphs_hash.append(h)
    return topK_subgraphs, topK_subgraphs_hash

def build_ground_truth2(data, subgraph_size, directed_flag=True):
    hash_to_info = {}
    # ground truth
    for graph_idx in tqdm.tqdm(range(len(data))):
        datapoint = data[graph_idx]
        n_node = datapoint.x.shape[0]
        adj = np.zeros((n_node, n_node))
        adj_attr = np.zeros((n_node, n_node))
        edge_index = datapoint.edge_index.cpu().tolist()
        edge_attr = datapoint.edge_attr.cpu().tolist()
        if directed_flag:
            level = datapoint.level.cpu().tolist()
        neighbors = [set() for _ in range(n_node)]
        for edge_idx in range(len(edge_index[0])):
            u = edge_index[0][edge_idx]
            v = edge_index[1][edge_idx]
            if not directed_flag or level[u] < level[v]:
                adj[u][v] = 1
                adj_attr[u][v] = np.argmax(np.array(edge_attr[edge_idx]))
                neighbors[u].add(v)
                neighbors[v].add(u)

        def backtrack(current, frontier, root, blocked):
            if len(current) == subgraph_size:
                G = networkx.DiGraph()
                sub_adj = adj[np.ix_(current, current)]
                sub_adj_attr = adj_attr[np.ix_(current, current)]
                sub_x_temp = datapoint.x[current]
                # sub_X = torch.argmax(datapoint.x[current], dim=1).cpu().tolist()
                sub_X = torch.argmax(sub_x_temp, dim=1).cpu().tolist() if sub_x_temp.dim() == 2 else sub_x_temp.cpu().tolist()
                for i in range(len(sub_X)):
                    G.add_node(i, operator = sub_X[i])
                for i in range(len(sub_X)):
                    for j in range(len(sub_X)):
                        if sub_adj[i][j] == 1:
                            G.add_edge(i, j, edge_type=sub_adj_attr[i][j])
                hash_G = networkx.weisfeiler_lehman_graph_hash(G, node_attr='operator', edge_attr='edge_type')

                if hash_G in hash_to_info.keys():
                    hash_to_info[hash_G].count += 1
                else:
                    sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                        subset=current,
                        edge_index=datapoint.edge_index,
                        edge_attr=datapoint.edge_attr,
                        relabel_nodes=True,
                        num_nodes=datapoint.x.shape[0]
                    )
                    sub_x = datapoint.x[current]
                    sub_y = datapoint.y
                    if directed_flag:
                        sub_level = realloc_subgraph_node_level(len(current), sub_edge_index, datapoint.level[current].numpy())
                        sub_level = torch.tensor(sub_level).float()
                        subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level, count=torch.tensor([1]).long().to(datapoint.x.device))
                    else:
                        subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, count=torch.tensor([1]).long().to(datapoint.x.device))
                    hash_to_info[hash_G] = subgraph_datapoint
                return
            old_frontier_set = set(frontier)
            for i, v in enumerate(frontier):
                new_current = current + [v]
                prefix = frontier[:i+1]
                for u in prefix:
                    blocked[u] = True

                tail_set = set(frontier[i+1:])
                new_neighbors = {
                    u for u in neighbors[v]
                    if (u > root) and (u not in new_current) and (not blocked[u])
                }

                new_frontier = sorted(tail_set | new_neighbors)
                backtrack(new_current, new_frontier, root, blocked)

                for u in prefix:
                    blocked[u] = False

        for root in range(n_node):
            current = [root]
            frontier = sorted([v for v in neighbors[root] if v > root])
            blocked = [False] * n_node
            backtrack(current, frontier, root, blocked)

    sorted_items = sorted(hash_to_info.items(), key=lambda kv: kv[1].count.item(), reverse=True)
    topK_subgraphs = [info for h, info in sorted_items]
    topK_subgraphs_hash = [h for h, info in sorted_items]

    return topK_subgraphs, topK_subgraphs_hash


def NRS(data, subgraph_size, subgraph_dataset_size, directed_flag=True, non_isomorphism=True):
    """
    Sample subgraphs from original dataset using weighted NRS

    Parameters:
    data: List[torch_geometric.data.Data]
    k: int
    dataset_size: int
    
    Returns:
    List[torch_geometric.data.Data]: random collected induced subgraphs with k vertices 
    """
    
    weights = []
    graphs = []
    
    # Note: Assuming nodes in the DAG have been topologically sorted in previous preprocessing steps
    for graph_idx in tqdm.tqdm(range(len(data))):
        weights.append(data[graph_idx].edge_index.shape[1])
        G = networkx.DiGraph()
        n_node = data[graph_idx].x.shape[0]
        for node_idx in range(n_node):
            G.add_node(node_idx)
        edge_index = data[graph_idx].edge_index.cpu().tolist()
        if directed_flag:
            level = data[graph_idx].level.cpu().tolist()
        for edge_idx in range(len(edge_index[0])):
            u = edge_index[0][edge_idx]
            v = edge_index[1][edge_idx]
            if not directed_flag or int(level[u]) < int(level[v]):
                G.add_edge(u, v)

        graphs.append(G)

    #training dataset
    random.seed(RANDOM_SEED)
    subgraph_number = 0
    subgraphs_data = []
    graph_indices = list(range(len(data)))
    cum_weights = np.cumsum(np.array(weights))
    hash_G_set = set()
    with tqdm.tqdm(total=subgraph_dataset_size) as pbar:
        while subgraph_number < subgraph_dataset_size:
            graph_idx = random.choices(graph_indices, cum_weights=cum_weights, k=1)[0]
            datapoint = data[graph_idx]
            G = graphs[graph_idx]
            sub_G = neighbour_reservoir_sampling(G, subgraph_size)
            sub_nodes = list(sub_G.nodes())
            sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                subset=sub_nodes,
                edge_index=datapoint.edge_index,
                edge_attr=datapoint.edge_attr,
                relabel_nodes=True,
                num_nodes=datapoint.x.shape[0]
            )
            sub_x = datapoint.x[sub_nodes]
            sub_y = datapoint.y
            if directed_flag:
                sub_level = realloc_subgraph_node_level(len(sub_nodes), sub_edge_index.cpu().numpy(), datapoint.level[sub_nodes].cpu().numpy())
                sub_level = torch.tensor(sub_level).float()
                subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level)
            else:
                subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y)
            subgraphs_data.append(subgraph_datapoint)
            # sub_X = torch.argmax(sub_x, dim=1).cpu().tolist()
            sub_X = torch.argmax(sub_x, dim=-1).cpu().tolist() if sub_x.dim() == 2 else sub_x.cpu().tolist()
            sub_edge_index = sub_edge_index.cpu().tolist()
            sub_edge_attr = torch.argmax(sub_edge_attr, dim=1).cpu().tolist()

            if directed_flag:
                hash_G = graph_hash(sub_X, sub_edge_index, sub_edge_attr, True, sub_level.numpy())
            else:
                hash_G = graph_hash(sub_X, sub_edge_index, sub_edge_attr, False, None)
            if non_isomorphism:
                if hash_G not in hash_G_set:
                    hash_G_set.add(hash_G)
                    subgraph_number += 1
                    pbar.update(1)
            else:
                subgraph_number += 1
                pbar.update(1)
    return subgraphs_data

def ARS(data, subgraph_size, subgraph_dataset_size, directed_flag=True, non_isomorphism=True):
    """
    Sample subgraphs from original dataset using weighted ARS

    Parameters:
    data: List[torch_geometric.data.Data]
    k: int
    dataset_size: int
    
    Returns:
    List[torch_geometric.data.Data]: random collected induced subgraphs with k vertices 
    """
    
    weights = []
    graphs = []
    
    # Note: Assuming nodes in the DAG have been topologically sorted in previous preprocessing steps
    for graph_idx in tqdm.tqdm(range(len(data))):
        weights.append(data[graph_idx].edge_index.shape[1])
        G = networkx.DiGraph()
        n_node = data[graph_idx].x.shape[0]
        for node_idx in range(n_node):
            G.add_node(node_idx)
        edge_index = data[graph_idx].edge_index.cpu().tolist()
        if directed_flag:
            level = data[graph_idx].level.cpu().tolist()
        for edge_idx in range(len(edge_index[0])):
            u = edge_index[0][edge_idx]
            v = edge_index[1][edge_idx]
            if not directed_flag or int(level[u]) < int(level[v]):
                G.add_edge(u, v)

        graphs.append(G)

    #training dataset
    random.seed(RANDOM_SEED)
    subgraph_number = 0
    subgraphs_data = []
    graph_indices = list(range(len(data)))
    cum_weights = np.cumsum(np.array(weights))
    hash_G_set = set()
    with tqdm.tqdm(total=subgraph_dataset_size) as pbar:
        while subgraph_number < subgraph_dataset_size:
            graph_idx = random.choices(graph_indices, cum_weights=cum_weights, k=1)[0]
            datapoint = data[graph_idx]
            n_node = datapoint.x.shape[0]
            lst = list(range(n_node))            
            lst_comb = list(combinations(lst, subgraph_size))
            random.shuffle(lst_comb)
            if len(lst_comb) == 0:
                continue
            sub_lst = list(lst_comb[0])
            
            adj = np.zeros((n_node, n_node))
            adj_attr = np.zeros((n_node, n_node))
            edge_index = datapoint.edge_index.cpu().tolist()
            edge_attr = datapoint.edge_attr.cpu().tolist()
            if directed_flag:
                level = datapoint.level.cpu().tolist()
            for edge_idx in range(len(edge_index[0])):
                u = edge_index[0][edge_idx]
                v = edge_index[1][edge_idx]
                if not directed_flag or level[u] < level[v]:
                    adj[u][v] = 1
                    adj_attr[u][v] = np.argmax(np.array(edge_attr[edge_idx]))
                
            bfs_flags = deque([sub_lst[0]])
            bfs_queues = deque([sub_lst[0]])

            while len(bfs_queues) > 0:
                size = len(bfs_queues)
                while size > 0:
                    u = bfs_queues.popleft()
                    for v in sub_lst:
                        if u != v and (adj[u][v] == 1 or adj[v][u] == 1):
                            if v not in bfs_flags:
                                bfs_flags.append(v)
                                bfs_queues.append(v)
                    size -= 1
            
            if len(bfs_flags) == len(sub_lst):
                sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                    subset=sub_lst,
                    edge_index=datapoint.edge_index,
                    edge_attr=datapoint.edge_attr,
                    relabel_nodes=True,
                    num_nodes=datapoint.x.shape[0]
                )
                sub_x = datapoint.x[sub_lst]
                sub_y = datapoint.y
                if directed_flag:
                    sub_level = realloc_subgraph_node_level(len(sub_lst), sub_edge_index, datapoint.level[sub_lst].numpy())
                    sub_level = torch.tensor(sub_level).float()
                    subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level)
                else:
                    subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y)

                subgraphs_data.append(subgraph_datapoint)

                # sub_X = torch.argmax(sub_x, dim=1).cpu().tolist()
                sub_X = torch.argmax(sub_x, dim=-1).cpu().tolist() if sub_x.dim() == 2 else sub_x.cpu().tolist()
                sub_edge_index = sub_edge_index.cpu().tolist()
                sub_edge_attr = torch.argmax(sub_edge_attr, dim=1).cpu().tolist()

                if directed_flag:
                    hash_G = graph_hash(sub_X, sub_edge_index, sub_edge_attr, True, sub_level.numpy())
                else:
                    hash_G = graph_hash(sub_X, sub_edge_index, sub_edge_attr, False, None)
                if non_isomorphism:
                    if hash_G not in hash_G_set:
                        hash_G_set.add(hash_G)
                        subgraph_number += 1
                        pbar.update(1)
                else:
                    subgraph_number += 1
                    pbar.update(1)

    return subgraphs_data

def Rand_ESU(data, subgraph_size, subgraph_dataset_size, tau=1, directed_flag=True, non_isomorphism=True, use_N_max=False):
    if use_N_max:
        N_max = subgraph_dataset_size // len(data) # max samples for one graph in one single round
    else:
        N_max = 10**100

    graph_indices = list(range(len(data)))
    subgraph_number = 0
    candidates = []
    seen = set()
    hash_G_list = list()
    p_list = [(1 - i / (subgraph_size + 1)) ** tau for i in range(1, subgraph_size+1)]

    random.seed(RANDOM_SEED)
    while subgraph_number < subgraph_dataset_size:
        random.shuffle(graph_indices)
        for graph_idx in tqdm.tqdm(graph_indices):
            # for save memory, calculate G every round
            datapoint = data[graph_idx]
            n_node = datapoint.x.shape[0]
            G = tensor_to_networkx(datapoint, directed_flag=directed_flag)
            neighbors = [set() for _ in range(n_node)]
            for (u, v) in G.edges():
                neighbors[u].add(v)
                neighbors[v].add(u)
            n_node = G.number_of_nodes()        
            n_samples = 0

            def handle_subgraph(V_subgraph):
                nonlocal subgraph_number
                nonlocal n_samples
                candidates.append({'graph_idx': graph_idx, 'S': list(V_subgraph)})
                sub_G = G.subgraph(V_subgraph)
                hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
                
                hash_G_list.append(hash_sub_G)

                if non_isomorphism:
                    if hash_sub_G not in seen:
                        seen.add(hash_sub_G)
                        subgraph_number += 1
                else:
                    subgraph_number += 1
                
                n_samples += 1
                if n_samples > N_max:
                    raise RuntimeError(f"n_samples ({n_samples}) > N_max ({N_max})")

            def extend_subgraph(V_subgraph, V_extension, root, L):
                if len(V_subgraph) == subgraph_size:
                    handle_subgraph(V_subgraph)
                    return
                while len(V_extension) != 0:
                    w = V_extension.pop()
                    exclusion_set = set()
                    for v in V_subgraph:
                        exclusion_set.update(neighbors[v])
                    new_extension = set([
                        u for u in neighbors[w]
                        if u > root 
                        and u not in exclusion_set
                    ])
                    V_extension_ = V_extension | new_extension
                    r = random.random()
                    if r < p_list[L]:
                        extend_subgraph(V_subgraph | {w}, V_extension_, root, L + 1)
            
            try:
                for root in range(n_node):
                    V_extension = set([u for u in neighbors[root] if u > root])
                    r = random.random()
                    if r < p_list[0]:
                        extend_subgraph({root}, V_extension, root, 1)
            except RuntimeError:
                pass
        print(f"subgraph_number/subgraph_dataset_size={subgraph_number}/{subgraph_dataset_size}")

    idxs = list(range(len(candidates)))
    random.shuffle(idxs)
    selected = []
    seen = set()
    for i in tqdm.tqdm(idxs):
        graph_idx = candidates[i]['graph_idx']
        datapoint = data[graph_idx]
        subset = list(candidates[i]['S'])
        sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
            subset=subset,
            edge_index=datapoint.edge_index,
            edge_attr=datapoint.edge_attr,
            relabel_nodes=True,
            num_nodes=datapoint.x.shape[0]
        )
        sub_x = datapoint.x[subset]
        sub_y = datapoint.y
        if directed_flag:
            sub_level = realloc_subgraph_node_level(len(subset), sub_edge_index, datapoint.level[subset].numpy())
            sub_level = torch.tensor(sub_level).float()
            subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level)
        else:
            subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y)
        selected.append(subgraph_datapoint)
        
        if non_isomorphism:
            seen.add(hash_G_list[i])
            if len(seen) == subgraph_dataset_size:
                break
        else:
            if len(selected) == subgraph_dataset_size:
                break
    
    return selected

def _rand_esu_round_worker(data, subgraph_size, subgraph_dataset_size, tau, directed_flag, non_isomorphism, out_q, progress_q=None, use_N_max=False):
    if use_N_max:
        N_max = subgraph_dataset_size // len(data) # max samples for one graph in one single round
    else:
        N_max = 10**100

    graph_indices = list(range(len(data)))
    subgraph_number = 0
    candidates = []
    seen = set()
    hash_G_list = list()
    p_list = [(1 - i / (subgraph_size + 1)) ** tau for i in range(1, subgraph_size+1)]

    random.shuffle(graph_indices)
    for graph_idx in graph_indices:
        # for save memory, calculate G every round
        datapoint = data[graph_idx]
        n_node = datapoint.x.shape[0]
        G = tensor_to_networkx(datapoint, directed_flag=directed_flag)
        neighbors = [set() for _ in range(n_node)]
        for (u, v) in G.edges():
            neighbors[u].add(v)
            neighbors[v].add(u)
        n_node = G.number_of_nodes()
        n_samples = 0

        def handle_subgraph(V_subgraph):
            nonlocal subgraph_number
            nonlocal n_samples
            candidates.append({'graph_idx': graph_idx, 'S': list(V_subgraph)})
            sub_G = G.subgraph(V_subgraph)
            hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
            
            hash_G_list.append(hash_sub_G)

            if non_isomorphism:
                if hash_sub_G not in seen:
                    seen.add(hash_sub_G)
                    subgraph_number += 1
            else:
                subgraph_number += 1
            
            n_samples += 1
            if n_samples > N_max:
                raise RuntimeError(f"n_samples ({n_samples}) > N_max ({N_max})")

        def extend_subgraph(V_subgraph, V_extension, root, L):
            if len(V_subgraph) == subgraph_size:
                handle_subgraph(V_subgraph)
                return
            while len(V_extension) != 0:
                w = V_extension.pop()
                exclusion_set = set()
                for v in V_subgraph:
                    exclusion_set.update(neighbors[v])
                new_extension = set([
                    u for u in neighbors[w]
                    if u > root 
                    and u not in exclusion_set
                ])
                V_extension_ = V_extension | new_extension
                r = random.random()
                if r < p_list[L]:
                    extend_subgraph(V_subgraph | {w}, V_extension_, root, L + 1)
        
        try:
            for root in range(n_node):
                V_extension = set([u for u in neighbors[root] if u > root])
                r = random.random()
                if r < p_list[0]:
                    extend_subgraph({root}, V_extension, root, 1)
        except RuntimeError:
            pass
        if progress_q is not None:
            progress_q.put(1)
    out_q.put((candidates, hash_G_list, subgraph_number))

def Rand_ESU_mp(data, subgraph_size, subgraph_dataset_size, tau=1, directed_flag=True, non_isomorphism=True, num_workers=None, use_N_max=False):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    elif num_workers >= mp.cpu_count() - 1:
        num_workers = mp.cpu_count() - 1
    total_candidates = []
    total_hashes = []
    total_count = 0
    random.seed(RANDOM_SEED)
    while total_count < subgraph_dataset_size:
        out_q = mp.Queue()
        progress_q = mp.Queue()
        procs = []
        for _ in range(num_workers):
            p = mp.Process(target=_rand_esu_round_worker, args=(data, subgraph_size, subgraph_dataset_size, tau, directed_flag, non_isomorphism, out_q, progress_q, use_N_max))
            p.start()
            procs.append(p)
        total_progress = num_workers * len(data)
        with tqdm.tqdm(total=total_progress, desc="RAND-ESU(mp)-round") as pbar:
            for _ in range(total_progress):
                progress_q.get()
                pbar.update(1)
        round_candidates = []
        round_hashes = []
        round_count = 0
        for _ in range(num_workers):
            cands, hashes, c = out_q.get()
            round_candidates.extend(cands)
            round_hashes.extend(hashes)
            round_count += c
        for p in procs:
            p.join()
        total_candidates.extend(round_candidates)
        total_hashes.extend(round_hashes)
        total_count += round_count
        print(f"{total_count}/{subgraph_dataset_size}")
        if total_count >= subgraph_dataset_size:
            break
    
    idxs = list(range(len(total_candidates)))
    random.shuffle(idxs)
    selected = []
    seen = set()
    for i in tqdm.tqdm(idxs):
        graph_idx = total_candidates[i]['graph_idx']
        datapoint = data[graph_idx]
        subset = list(total_candidates[i]['S'])
        sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
            subset=subset,
            edge_index=datapoint.edge_index,
            edge_attr=datapoint.edge_attr,
            relabel_nodes=True,
            num_nodes=datapoint.x.shape[0]
        )
        sub_x = datapoint.x[subset]
        sub_y = datapoint.y
        if directed_flag:
            sub_level = realloc_subgraph_node_level(len(subset), sub_edge_index, datapoint.level[subset].numpy())
            sub_level = torch.tensor(sub_level).float()
            subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y, level=sub_level)
        else:
            subgraph_datapoint = torch_geometric.data.Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=sub_y)
        selected.append(subgraph_datapoint)
        if non_isomorphism:
            seen.add(total_hashes[i])
            if len(seen) == subgraph_dataset_size:
                break
        else:
            if len(selected) == subgraph_dataset_size:
                break
    return selected




def Rand_GTRIE(
    data,
    subgraph_size,
    subgraph_dataset_size,
    directed_flag=True,
    non_isomorphism=False,
    tau=1.0,
    cache_size=64,
    batch_size=512,
    max_rounds_without_gain=5,
):
    if batch_size is None or batch_size <= 0:
        batch_size = len(data)
    gtrie_indices = list(range(len(data)))
    batches = [gtrie_indices[i:i+batch_size] for i in range(0, len(gtrie_indices), batch_size)]
    rng = random.Random(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    hash_G_set = set()
    selected = []
    selected_hashes = []
    frequency_estimates = defaultdict(float)
    p_list = [(1 - i / (subgraph_size + 1)) ** tau for i in range(1, subgraph_size + 1)]
    repr_cache = GraphRepresentationCache(data, directed_flag, max_size=cache_size)

    def current_count():
        return len(hash_G_set) if non_isomorphism else len(selected)

    for batch_idx, batch in enumerate(batches):
        if current_count() >= subgraph_dataset_size:
            break
        batch_data = [data[i] for i in batch]
        gtrie = construct_gtrie(batch_data, subgraph_size, directed_flag)
        if not gtrie.has_patterns():
            continue
        rounds_without_gain = 0
        while current_count() < subgraph_dataset_size and rounds_without_gain <= max_rounds_without_gain:
            random.shuffle(batch)
            progress = False
            for graph_idx in tqdm.tqdm(batch, desc=f"batch {batch_idx+1}/{len(batches)} (k={subgraph_size})"):
                if current_count() >= subgraph_dataset_size:
                    break
                datapoint = data[graph_idx]
                adj_matrix, node_labels = repr_cache.get(graph_idx)
                sampled_any = False
                for subset, occ_prob in gtrie.sample_matches(adj_matrix, node_labels, p_list, rng):
                    if occ_prob is None or occ_prob <= 0:
                        continue
                    subset_sorted = sorted(subset)
                    subgraph_datapoint, hash_G = extract_subgraph(datapoint, subset_sorted, directed_flag, materialize_data=True)
                    weight = 1.0 / max(occ_prob, 1e-12)
                    subgraph_datapoint.sample_prob = torch.tensor([occ_prob], dtype=torch.float32)
                    subgraph_datapoint.sample_weight = torch.tensor([weight], dtype=torch.float32)
                    frequency_estimates[hash_G] += weight
                    if non_isomorphism:
                        if hash_G in hash_G_set:
                            continue
                        hash_G_set.add(hash_G)
                    selected.append(subgraph_datapoint)
                    selected_hashes.append(hash_G)
                    sampled_any = True
                    progress = True
                    if current_count() >= subgraph_dataset_size:
                        break
                if current_count() >= subgraph_dataset_size:
                    break
                if not sampled_any:
                    continue
            if not progress:
                warnings.warn("Rand_GTRIE stopped early within batch because insufficient unique subgraphs were found.")
                rounds_without_gain += 1
            else:
                rounds_without_gain = 0
            print(f"{current_count()}/{subgraph_dataset_size}")

    final_selected = selected[:subgraph_dataset_size]
    final_hashes = selected_hashes[:subgraph_dataset_size]
    for data_obj, h in zip(final_selected, final_hashes):
        est = frequency_estimates.get(h)
        if est is not None:
            data_obj.estimated_count = torch.tensor([est], dtype=torch.float32)
    return final_selected


def _merge_freq_maps(freq_maps):
    merged = {}
    for fm in freq_maps:
        for key, record in fm.items():
            if key not in merged:
                merged[key] = {
                    "estimated_count": record.get("estimated_count", 0.0),
                    "occurrences": list(record.get("occurrences", [])),
                }
            else:
                merged[key]["estimated_count"] += record.get("estimated_count", 0.0)
                merged[key]["occurrences"].extend(record.get("occurrences", []))
    return merged


def _finalize_fase_samples(freq_map, data, subgraph_dataset_size, directed_flag, non_isomorphism, rng):
    def _unwrap_occ(occ):
        if isinstance(occ, tuple):
            g, nodes, sample_prob, sample_weight = occ
            return int(g), list(nodes), float(sample_prob), float(sample_weight)
        return int(occ.graph_idx), list(occ.nodes), float(occ.sample_prob), float(occ.sample_weight)

    if non_isomorphism:
        occurrence_pool = []
        for record in freq_map.values():
            est = record.get("estimated_count", 0.0)
            for occ in record.get("occurrences", []):
                occurrence_pool.append((occ, est))
        if not occurrence_pool:
            return []
        rng.shuffle(occurrence_pool)
        seen_hashes = set()
        samples = []
        for occ, est in tqdm.tqdm(occurrence_pool, desc="Rand_FaSE finalize (unique categories)", total=len(occurrence_pool)):
            if len(seen_hashes) >= subgraph_dataset_size:
                break
            graph_idx, nodes, sample_prob, sample_weight = _unwrap_occ(occ)
            datapoint = data[graph_idx]
            subset = list(nodes)
            subgraph_datapoint, hash_G = extract_subgraph(
                datapoint, subset, directed_flag, materialize_data=True
            )
            if hash_G in seen_hashes:
                pass
            else:
                seen_hashes.add(hash_G)
            subgraph_datapoint.sample_prob = torch.tensor([sample_prob], dtype=torch.float32)
            subgraph_datapoint.sample_weight = torch.tensor([sample_weight], dtype=torch.float32)
            subgraph_datapoint.estimated_count = torch.tensor([est], dtype=torch.float32)
            samples.append(subgraph_datapoint)
        return samples

    occurrence_pool = []
    for record in freq_map.values():
        occurrences = record.get("occurrences", [])
        if not occurrences:
            continue
        for occ in occurrences:
            occurrence_pool.append((occ, record["estimated_count"]))
    if not occurrence_pool:
        return []
    rng.shuffle(occurrence_pool)
    selected_occurrences = occurrence_pool[:subgraph_dataset_size]

    samples = []
    for occ, record_estimate in tqdm.tqdm(selected_occurrences, desc="Rand_FaSE finalize (samples)", total=len(selected_occurrences)):
        datapoint = data[occ.graph_idx]
        subset = list(occ.nodes)
        subgraph_datapoint, _ = extract_subgraph(
            datapoint, subset, directed_flag, materialize_data=True
        )
        subgraph_datapoint.sample_prob = torch.tensor([occ.sample_prob], dtype=torch.float32)
        subgraph_datapoint.sample_weight = torch.tensor([occ.sample_weight], dtype=torch.float32)
        subgraph_datapoint.estimated_count = torch.tensor([record_estimate], dtype=torch.float32)
        samples.append(subgraph_datapoint)
    return samples


def _rand_fase_round_worker(
    data,
    subgraph_size,
    subgraph_dataset_size,
    directed_flag,
    non_isomorphism,
    tau,
    seed,
    use_N_max,
    out_q,
    progress_q=None,
):
    result = Rand_FaSE(
        data,
        subgraph_size,
        subgraph_dataset_size,
        directed_flag,
        non_isomorphism,
        tau,
        seed,
        use_N_max,
        parallel_splits=1,
        progress_q=progress_q,
        return_freq_map_only=True,
    )
    out_q.put(result)
    if progress_q is not None:
        progress_q.put(None)


def Rand_FaSE(
    data,
    subgraph_size,
    subgraph_dataset_size,
    directed_flag=True,
    non_isomorphism=False, 
    tau=3.0,
    seed=RANDOM_SEED,
    use_N_max=True,
    parallel_splits=4,
    progress_q=None,
    return_freq_map_only=False,
):
    assert not (
        non_isomorphism and parallel_splits is not None and parallel_splits > 1
    ), "Rand_FaSE does not support non_isomorphism=True with parallel_splits>1."

    if parallel_splits is not None and parallel_splits > 1:
        num_workers = parallel_splits
        if num_workers >= mp.cpu_count() - 1:
            num_workers = max(1, mp.cpu_count() - 1)
        per_target = math.ceil(subgraph_dataset_size / num_workers)
        base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        out_q = mp.Queue()
        progress_q = mp.Queue()
        procs = []
        for worker_id in range(num_workers):
            worker_seed = base_seed + worker_id
            p = mp.Process(
                target=_rand_fase_round_worker,
                args=(
                    data,
                    subgraph_size,
                    per_target,
                    directed_flag,
                    non_isomorphism,
                    tau,
                    worker_seed,
                    use_N_max,
                    out_q,
                    progress_q,
                ),
            )
            p.start()
            procs.append(p)
        total_progress = num_workers * len(data)
        with tqdm.tqdm(total=total_progress, desc="Rand_FaSE(mp)-graphs") as pbar:
            done_workers = 0
            while done_workers < num_workers:
                msg = progress_q.get()
                if msg is None:
                    done_workers += 1
                    continue
                pbar.update(1)
                
        results = []
        for _ in range(num_workers):
            results.append(out_q.get())
        for p in procs:
            p.join()
        merged_freq = _merge_freq_maps(results)
        if non_isomorphism:
            current_output = len(merged_freq)
        else:
            current_output = sum(len(record.get("occurrences", [])) for record in merged_freq.values())
        print(f"Rand_FaSE(mp) collected {min(current_output, subgraph_dataset_size)}/{subgraph_dataset_size} subgraphs")
        rng = random.Random(seed)
        return _finalize_fase_samples(
            merged_freq,
            data,
            subgraph_dataset_size,
            directed_flag,
            non_isomorphism,
            rng,
        )

    rng = random.Random(seed)
    neighbour_cache = build_neighbour_cache(data, directed_flag)
    engine = FaSEEngine(subgraph_size, tau, directed_flag)

    graph_indices = list(range(len(data)))
    per_graph_quota = None
    if use_N_max and len(graph_indices) > 0:
        per_graph_quota = max(1, math.ceil(subgraph_dataset_size / len(graph_indices)))
    graphs_processed = 0
    current_output = 0
    freq_map = {}

    def update_progress(round_idx):
        graphs_total = max(1, round_idx * len(graph_indices))
        graphs_desc = f"{min(graphs_processed, graphs_total)}/{graphs_total} graphs"
        if non_isomorphism:
            subgraph_count = min(current_output, subgraph_dataset_size)
        else:
            subgraph_count = min(int(current_output), subgraph_dataset_size)
        subgraphs_desc = f"{subgraph_count}/{subgraph_dataset_size} subgraphs"
        print(f"Rand_FaSE round={round_idx} | {graphs_desc}: {subgraphs_desc}")

    round_idx = 0
    if progress_q is None:
        update_progress(round_idx)

    prev_output = -1
    while True:
        round_idx += 1
        rng.shuffle(graph_indices)
        round_start = time.time()
        graphs_processed_round = 0
        if progress_q is None:
            with tqdm.tqdm(
                graph_indices,
                desc=f"Rand_FaSE round={round_idx} graph progress",
            ) as graph_pbar:
                for graph_idx in graph_pbar:
                    adj_map, neighbors, node_labels = neighbour_cache.get(graph_idx)
                    engine.process_graph(
                        graph_idx,
                        adj_map,
                        neighbors,
                        node_labels,
                        rng,
                        max_samples_per_graph=per_graph_quota,
                    )
                    graphs_processed += 1
                    graphs_processed_round += 1
                    elapsed = max(1e-6, time.time() - round_start)
                    remaining = len(graph_indices) - graphs_processed_round
                    eta = remaining * (elapsed / graphs_processed_round)
                    graph_pbar.set_postfix_str(f"{graphs_processed_round}/{len(graph_indices)} graphs | ETA: {eta:.1f}s")
        else:
            for graph_idx in graph_indices:
                adj_map, neighbors, node_labels = neighbour_cache.get(graph_idx)
                engine.process_graph(
                    graph_idx,
                    adj_map,
                    neighbors,
                    node_labels,
                    rng,
                    max_samples_per_graph=per_graph_quota,
                )
                graphs_processed += 1
                graphs_processed_round += 1
                progress_q.put(1)

        freq_map = engine.collect_results()
        if non_isomorphism:
            current_output = len(freq_map)
        else:
            current_output = sum(len(record.get("occurrences", [])) for record in freq_map.values())
        print(f"Rand_FaSE(mp) collected {min(current_output, subgraph_dataset_size)}/{subgraph_dataset_size} subgraphs")
        if progress_q is None:
            update_progress(round_idx)

        if current_output >= subgraph_dataset_size:
            break
        if current_output == prev_output:
            warnings.warn("Rand_FaSE stopped early: no new subgraphs collected in this round.")
            break
        prev_output = current_output

    if return_freq_map_only:
        return freq_map

    return _finalize_fase_samples(
        freq_map,
        data,
        subgraph_dataset_size,
        directed_flag,
        non_isomorphism,
        rng,
    )


if __name__ == "__main__":
    from grade.datasets.loader.Younger import YoungerDataset
    import time
    from datetime import datetime
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root = pathlib.Path(cur_dir) / ".." / ".." / ".." / "data" / "Younger"
    logs_dir = pathlib.Path(cur_dir) / ".." / ".." / ".." / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    for ss in [5]:
        for ssm in ["RESU", "NRS"]:
            log_path = logs_dir / f"{ssm}-{ss}.log"
            start_ts = time.time()
            start_dt = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")

            dataset = YoungerDataset(root, "train", ssm, ss, 40000)
            print(dataset.get_dataset_info())
            dataset[0]

            end_ts = time.time()
            end_dt = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")
            duration = end_ts - start_ts

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"start_ts: {start_ts}\n")
                f.write(f"start_time: {start_dt}\n")
                f.write(f"end_ts: {end_ts}\n")
                f.write(f"end_time: {end_dt}\n")
                f.write(f"duration: {duration}\n")
                f.write("------------------------------\n")
