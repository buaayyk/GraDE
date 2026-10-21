import argparse
import pathlib
import os
import json
import numpy as np
import pandas as pd

from grade.datasets.loader.NASBench101 import NASBench101Dataset
from grade.datasets.loader.NASBench201 import NASBench201Dataset
from grade.datasets.loader.NASBench301 import NASBench301Dataset
from grade.datasets.loader.NASBenchNLP import NASBenchNLPDataset
from grade.datasets.loader.Younger import YoungerDataset
from grade.datasets.loader.YoungerPart import YoungerPartDataset
from grade.datasets.loader.PROTEINS import PROTEINSDataset
from grade.datasets.loader.SocioPatternsHighSchool import SocioPatternsHighSchoolDataset
from grade.evaluation.metrics import (
    kendall_correlation,
    spearman_correlation,
)
from grade.evaluation.sampling_only import get_estimate_values
from grade.evaluation.discovery import set_discovery_args

cur_dir = os.path.dirname(os.path.abspath(__file__))

def evaluation_run(args: argparse.Namespace):
    if args.sampling_only == 0 and not pathlib.Path(args.estimate_logps_path).exists():
        raise FileNotFoundError(args.estimate_logps_path)
    if args.dataset == "NAS-Bench-101":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBench101"
        train_set = NASBench101Dataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBench101Dataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-201":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBench201"
        train_set = NASBench201Dataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBench201Dataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-301":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBench301"
        train_set = NASBench301Dataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBench301Dataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-NLP":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBenchNLP"
        train_set = NASBenchNLPDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBenchNLPDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "Younger":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "Younger"
        train_set = YoungerDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = YoungerDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "YoungerPart":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "YoungerPart"
        train_set = YoungerPartDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = YoungerPartDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "PROTEINS":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "PROTEINS"
        train_set = PROTEINSDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = PROTEINSDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "SocioPatterns-HighSchool":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "SocioPatternsHighSchool"
        train_set = SocioPatternsHighSchoolDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = SocioPatternsHighSchoolDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    else:
        raise NotImplementedError
    
    topK_subgraphs_count = []
    for idx in range(len(estimate_set)):
        topK_subgraphs_count.append(estimate_set[idx].count.cpu().item())

    if args.sampling_only == 1:
        estimate_values = get_estimate_values(train_set, estimate_set)
        estimate_values = {"G_logp": np.array(estimate_values)[:, None]}
    else:
        with open(args.estimate_logps_path, "r") as f:
            estimate_values = json.load(f)
    for name, values in estimate_values.items():
        arr = np.array(values)
        rounds = arr.shape[-1]
        metric_to_series = {metric: [] for metric in args.metrics}
        for r in range(rounds):
            avg_values = np.mean(arr[:, :r+1], axis=-1)
            for metric in args.metrics:
                if metric == 'spearman':
                    correlation, _ = spearman_correlation(avg_values, topK_subgraphs_count)
                    metric_to_series[metric].append(correlation)
                elif metric == 'kendall':
                    correlation, _ = kendall_correlation(avg_values, topK_subgraphs_count)
                    metric_to_series[metric].append(correlation)
                else:
                    raise NotImplementedError
        if args.results_dirpath is not None:
            results_dirpath = pathlib.Path(args.results_dirpath)
            results_dirpath.mkdir(parents=True, exist_ok=True)
            result_path = results_dirpath / f"{name}.csv"
            df = pd.DataFrame(
                {metric: metric_to_series[metric] for metric in args.metrics},
                index=np.arange(1, rounds + 1)
            )
            df.index.name = "rounds"
            df.to_csv(result_path)
        else:
            print(name)
            for r in range(rounds):
                summary = " ".join(
                    f"{metric_to_series[metric][r]:.4f}" 
                    for metric in args.metrics
                )
                print(r + 1, summary, end=" ")
            print()

def set_model_evaluation_args(subparsers: argparse._SubParsersAction):
    evaluation_parser: argparse.ArgumentParser = subparsers.add_parser('model')
    evaluation_parser.add_argument('--sampling_only', type=int, default=0)
    evaluation_parser.add_argument('--estimate_logps_path', type=str)
    evaluation_parser.add_argument('--dataset', type=str, choices=["NAS-Bench-101", "NAS-Bench-201", "NAS-Bench-301", "NAS-Bench-NLP", "Younger", "YoungerPart", "PROTEINS", "SocioPatterns-HighSchool"], default="NAS-Bench-101")
    evaluation_parser.add_argument('-ssm', '--subgraph_sampling_method', type=str, choices=[None, "NRS", "ARS", "RESU", "RFASE"], default=None)
    evaluation_parser.add_argument('-ss', '--subgraph_size', type=int, default=None)
    evaluation_parser.add_argument('-sd', '--sampling_density', type=float, default=None)
    evaluation_parser.add_argument('--metrics', type=str, nargs='+', choices=['spearman', 'kendall'], default=['spearman'])
    evaluation_parser.add_argument('--results_dirpath', type=str, default=None)
    evaluation_parser.set_defaults(run=evaluation_run)

def set_evaluation_args(subparsers: argparse._SubParsersAction):
    evaluation_parser = subparsers.add_parser('evaluation')
    evaluation_subparsers = evaluation_parser.add_subparsers(required=True)
    set_model_evaluation_args(evaluation_subparsers)
    set_discovery_args(evaluation_subparsers)
