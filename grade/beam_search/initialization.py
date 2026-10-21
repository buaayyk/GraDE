import torch
import os
import pathlib
import networkx
import json
from tqdm import tqdm
import argparse
from datetime import datetime

from grade.datasets.loader.Younger import YoungerDataset
from grade.utils.format_transform import networkx_to_tensor, tensor_to_networkx
from grade.beam_search.graph_store import GraphDatabase

def initialize(args):
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    root = pathlib.Path(cur_dir).parents[1] / "data" / "Younger"
    train_set = YoungerDataset(root, "train")
    val_set = YoungerDataset(root, "valid")
    test_set = YoungerDataset(root, "test")
    score_dataset = train_set + val_set + test_set

    example_data = score_dataset[0]
    n_node_type = example_data.x.shape[-1]
    n_edge_type = example_data.edge_attr.shape[-1]

    with open(root / f"all_{args.subgraph_size}.json", "r") as f:
        hash_to_info = json.load(f)

    graph_db = GraphDatabase(score_dataset, build_networkx=True)
    infos = list(hash_to_info.values())
    infos = sorted(infos, key=lambda x:x['count'], reverse=True)

    candidates = []
    candidates_tensor = []
    counts = []

    for idx in tqdm(range(args.topK)):
        info = infos[idx]
        sub_G = graph_db.query(info['graph_idx']).subgraph(info['S'])
        temp_list = []
        for _ in range(args.max_instance):
            result = graph_db.subgraph_match(sub_G)
            if result['match']:
                graph_idx = result['graph_idx']
                S = result['S']
                temp_list.append({'graph_idx': graph_idx, 'S': list(S)})    

        # filter the same instance
        seen = set()
        for new_info in temp_list:
            graph_idx = new_info['graph_idx']
            S_sorted_tuple = tuple(sorted(new_info['S']))
            seen.add((graph_idx, S_sorted_tuple))

        candidates.append([])
        for element in seen:
            candidates[-1].append({'graph_idx': element[0], 'S': list(element[1])})

        candidate_tensor = networkx_to_tensor(sub_G, n_node_type, n_edge_type)
        candidates_tensor.append(candidate_tensor)
        counts.append(info['count'])

    work_dirpath = root / f"{args.subgraph_size}-node-{args.subgraph_sampling_method}-{args.sampling_density}-{args.algo}"
    if args.work_dir is not None:
        work_dirpath = args.work_dir.resolve()

    work_dirpath.mkdir(parents=True, exist_ok=True)
    with open(work_dirpath / "candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    torch.save(candidates_tensor, work_dirpath / "candidates_tensor.pt")
    with open(root / work_dirpath / "probs.json", "w") as f:
        json.dump(counts, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Younger')
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--max_instance', type=int, default=10)
    parser.add_argument('-ss', '--subgraph_size', type=int, default=3)
    parser.add_argument('-ssm', "--subgraph_sampling_method", type=str, default="NRS")
    parser.add_argument('-sd', "--sampling_density", type=int, default=400000)
    parser.add_argument('--algo', type=str, default='DisCo')
    parser.add_argument('--work_dir', type=pathlib.Path, default=None)
    args = parser.parse_args()
    initialize(args)
