from grade.utils.random_seed import RANDOM_SEED

import torch
import os
import pathlib
import networkx
from tqdm import tqdm
import random

from grade.datasets.loader.Younger import YoungerDataset
from grade.utils.format_transform import networkx_to_tensor, tensor_to_networkx
import json
from grade.beam_search.graph_store import GraphDatabase
from grade.beam_search.selection import select_topk, select_indices

import argparse

def expand_one_node(G: networkx.DiGraph, S):
    added_nodes = set()
    for u in S:
        for v in G.predecessors(u):
            if v not in S:
                added_nodes.add(v)
        for v in G.successors(u):
            if v not in S:
                added_nodes.add(v)
    new_S_list = []
    for added_node in added_nodes:
        new_S_list.append(set(S) | {added_node})
    return new_S_list

def load_dataset(root):
    train_set = YoungerDataset(root, "train")
    val_set = YoungerDataset(root, "valid")
    test_set = YoungerDataset(root, "test")
    return train_set + val_set + test_set


def expand(args):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if args.topK < 1 or args.max_instance < 1:
        raise ValueError("topK and max_instance must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    root = pathlib.Path(cur_dir).parents[1] / "data" / args.dataset

    work_dirpath = root / f"{args.subgraph_size}-node-{args.subgraph_sampling_method}-{args.sampling_density}-{args.algo}-{args.topK}-{args.max_instance}-{args.selection}"
    work_dirpath2 = root / f"{args.subgraph_size+1}-node-{args.subgraph_sampling_method}-{args.sampling_density}-{args.algo}-{args.topK}-{args.max_instance}-{args.selection}"
    if args.work_dir is not None:
        work_dirpath = args.work_dir.resolve()
    if args.next_work_dir is not None:
        work_dirpath2 = args.next_work_dir.resolve()
    
    # step 1
    with open(work_dirpath / "candidates.json", "r") as f:
        candidates = json.load(f)
    with open(work_dirpath / "probs.json") as f:
        probs = json.load(f)

    # step 2
    if args.selection == 'topk':
        candidate_prob_pair = select_topk(candidates, probs)
    else:
        selected_indices = select_indices(
            probs,
            args.topK,
            args.temperature,
            direct_probabilities=args.subgraph_size == 3,
        )
        with open(work_dirpath / "selected.json", "w") as file:
            json.dump(selected_indices, file, indent=2)
        candidate_prob_pair = [(candidates[idx], probs[idx]) for idx in selected_indices]

    score_dataset = load_dataset(root)
    example_data = score_dataset[0]
    n_node_type = example_data.x.shape[-1]
    n_edge_type = example_data.edge_attr.shape[-1]

    # step 3
    graph_db = GraphDatabase(score_dataset, build_networkx=True)
    random.seed(args.seed)
    new_infos = []
    for idx in tqdm(range(min(len(candidates), args.topK))):
        candidate = candidate_prob_pair[idx][0]
        random.shuffle(candidate)
        for info_idx in range(min(len(candidate), args.max_instance)):
            info = candidate[info_idx]
            graph_idx = info['graph_idx']
            G = graph_db.query(graph_idx)
            S = info['S']
            new_S_list = expand_one_node(G, S)
            new_infos = new_infos + [{'graph_idx': graph_idx, 'S': list(new_S)} for new_S in new_S_list]

    # step 4
    hash_to_new_candidate = {}
    for idx in tqdm(range(len(new_infos))):
        new_info = new_infos[idx]
        graph_idx = new_info['graph_idx']
        G = graph_db.query(graph_idx)
        S = new_info['S']
        sub_G = G.subgraph(S)
        hash_sub_G = networkx.weisfeiler_lehman_graph_hash(sub_G, node_attr='operator', edge_attr='edge_type')
        if hash_sub_G not in hash_to_new_candidate.keys():
            hash_to_new_candidate[hash_sub_G] = []
        hash_to_new_candidate[hash_sub_G].append(new_info)

    # step 5: filter the same instances
    for k, new_candidate in hash_to_new_candidate.items():
        seen = set()
        for info in new_candidate:
            graph_idx = info['graph_idx']
            S_sorted_tuple = tuple(sorted(info['S']))
            seen.add((graph_idx, S_sorted_tuple))
        filtered_new_candidate = []
        for element in seen:
            filtered_new_candidate.append({'graph_idx': element[0], 'S': list(element[1])})
        hash_to_new_candidate[k] = filtered_new_candidate

    # step 6
    new_candidates = list(hash_to_new_candidate.values())
    new_candidates_tensor = []
    for idx in tqdm(range(len(new_candidates))):
        new_info = new_candidates[idx][0]
        graph_idx = new_info['graph_idx']
        G = graph_db.query(graph_idx)
        S = new_info['S']
        sub_G = G.subgraph(S)
        datapoint = networkx_to_tensor(sub_G, n_node_type, n_edge_type)
        new_candidates_tensor.append(datapoint)
    
    work_dirpath2.mkdir(parents=True, exist_ok=True)
    with open(work_dirpath2 / "candidates.json", "w") as f:
        json.dump(new_candidates, f, indent=2)
    torch.save(new_candidates_tensor, work_dirpath2 / "candidates_tensor.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Younger')
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--max_instance', type=int, default=20)
    parser.add_argument('-ss', '--subgraph_size', type=int, default=3)
    parser.add_argument('-ssm', "--subgraph_sampling_method", type=str, default="RESU")
    parser.add_argument('-sd', "--sampling_density", type=int, default=400000)
    parser.add_argument('--algo', type=str, default='DisCo')
    parser.add_argument('--selection', choices=['topk', 'stochastic'], default='topk')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--work_dir', type=pathlib.Path, default=None)
    parser.add_argument('--next_work_dir', type=pathlib.Path, default=None)
    args = parser.parse_args()
    expand(args)
