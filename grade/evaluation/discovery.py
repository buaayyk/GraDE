import torch
import os
import pathlib
import networkx
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
import json

from grade.datasets.loader.Younger import YoungerDataset
from grade.utils.format_transform import networkx_to_tensor, tensor_to_networkx

from grade.beam_search.graph_store import GraphDatabase

import argparse

def discovery_run(args):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    root = pathlib.Path(cur_dir).parents[1] / "data" / args.dataset
    train_set = YoungerDataset(root, "train")
    val_set = YoungerDataset(root, "valid")
    test_set = YoungerDataset(root, "test")
    score_dataset = train_set + val_set + test_set

    example_data = score_dataset[0]
    n_node_type = example_data.x.shape[-1]
    n_edge_type = example_data.edge_attr.shape[-1]

    work_dirpath = root / f"{args.subgraph_size}-node-{args.subgraph_sampling_method}-{args.sampling_density}-{args.algo}-{args.topK}-{args.max_instance}-{args.selection}"
    if args.work_dir is not None:
        work_dirpath = args.work_dir.resolve()
    work_dirpath.mkdir(parents=True, exist_ok=True)
    print(work_dirpath)

    with open(work_dirpath / "candidates.json", "r") as f:
        candidates = json.load(f)

    with open(work_dirpath / "probs.json", "r") as f:
        probs = json.load(f)

    idx_mapping = [(i, probs[i]) for i in range(len(probs))]
    idx_mapping = sorted(idx_mapping, key=lambda x: x[1], reverse=True)

    counts_path = work_dirpath / "counts.json"
    acounts_path = work_dirpath / "acounts.json"

    if args.resume and counts_path.exists():
        with open(counts_path, "r") as f:
            counts = json.load(f)
        print("Hello")
    else:
        counts = [-1 for _ in range(len(candidates))]
    acounts = [-1 for _ in range(len(candidates))]

    graph_db = GraphDatabase(score_dataset)
    for i in range(min(len(candidates), args.eval_topK)):
        idx = idx_mapping[i][0]
        info = candidates[idx][0]
        graph_idx = info['graph_idx']
        G = graph_db.query(graph_idx)
        S = info['S']
        sub_G = G.subgraph(S)
        
        # automorphism
        acount = graph_db.automorphism_count(sub_G)
        acounts[idx] = acount
        with open(acounts_path, "w") as f:
            json.dump(acounts, f, indent=2)

        if args.resume and counts[idx] != -1:
            t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            count = counts[idx]
            print(f"[{t}] {i}/{min(len(candidates), args.eval_topK)} {count}/{acount}={count//acount}")
            print("skip!")
            continue
        
        count = graph_db.subgraph_count_mp(sub_G, num_workers=10)

        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{t}] {i}/{min(len(candidates), args.eval_topK)} {count}/{acount}={count//acount}")
        
        counts[idx] = count
        with open(counts_path, "w") as f:
            json.dump(counts, f, indent=2)

def set_discovery_args(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('discovery')
    parser.add_argument('--dataset', type=str, default='Younger')
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--resume', action='store_true', help='Skip indices already having counts in counts.json')
    parser.add_argument('-ss', '--subgraph_size', type=int, default=3)
    parser.add_argument('-ssm', "--subgraph_sampling_method", type=str, default="NRS")
    parser.add_argument('-sd', "--sampling_density", type=int, default=400000)
    parser.add_argument('--algo', type=str, default='DisCo')
    parser.add_argument('--eval_topK', type=int, default=100)
    parser.add_argument('--max_instance', type=int, default=20)
    parser.add_argument('--selection', choices=['topk', 'stochastic'], default='topk')
    parser.add_argument('--work_dir', type=pathlib.Path, default=None)
    parser.set_defaults(run=discovery_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_discovery_args(parser.add_subparsers(required=True))
    args = parser.parse_args()
    args.run(args)
