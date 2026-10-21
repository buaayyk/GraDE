import argparse
from grade.datasets.loader.NASBench101 import NASBench101Dataset
from grade.datasets.loader.NASBench201 import NASBench201Dataset
from grade.datasets.loader.PROTEINS import PROTEINSDataset
from grade.datasets.loader.SocioPatternsHighSchool import SocioPatternsHighSchoolDataset
import os
import pathlib

def datasets_run(args: argparse.Namespace):
    cur_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    if args.dataset == "NAS-Bench-101":
        dataset_root = cur_dir / ".." / ".." / "data" / "NASBench101"
        dataset = NASBench101Dataset(dataset_root, args.split, args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-201":
        dataset_root = cur_dir / ".." / ".." / "data" / "NASBench201"
        dataset = NASBench201Dataset(dataset_root, args.split, args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "PROTEINS":
        dataset_root = cur_dir / ".." / ".." / "data" / "PROTEINS"
        dataset = PROTEINSDataset(dataset_root, args.split, args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "SocioPatterns-HighSchool":
        dataset_root = cur_dir / ".." / ".." / "data" / "SocioPatternsHighSchool"
        dataset = SocioPatternsHighSchoolDataset(dataset_root, args.split, args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    else:
        raise NotImplementedError(f"{args.dataset} hasn't been implemented!")

    if args.mode == "prepare":
        dataset.prepare()
    else:
        dataset_info = dataset.get_dataset_info()
        for k, v in dataset_info.items():
            print(f"{k}: {v}")
    
def set_datasets_args(subparsers: argparse._SubParsersAction):
    datasets_parser: argparse.ArgumentParser = subparsers.add_parser('datasets')
    datasets_parser.add_argument('--mode', type=str, choices=['prepare', 'show'], default='prepare')
    datasets_parser.add_argument('--dataset', type=str, choices=["NAS-Bench-101", "NAS-Bench-201", "PROTEINS", "SocioPatterns-HighSchool"], default="NAS-Bench-101")
    datasets_parser.add_argument('--split', type=str, choices=["train", "valid", "test"], default="train")
    datasets_parser.add_argument('-ssm', '--subgraph_sampling_method', type=str, choices=[None, "NRS", "ARS"], default=None)
    datasets_parser.add_argument('-ss', '--subgraph_size', type=int, default=None)
    datasets_parser.add_argument('-sd', '--sampling_density', type=float, default=None)
    datasets_parser.set_defaults(run=datasets_run)
