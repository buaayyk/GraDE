import argparse
import copy
import pathlib
import re

from grade.beam_search.aggregation import aggregate
from grade.beam_search.enumeration import enumerate_subgraphs
from grade.beam_search.expansion import expand
from grade.beam_search.initialization import initialize
from grade.diffusion.api import (
    set_DeFoG_args,
    set_DiGress_args,
    set_DisCo_args,
    set_GNNRanker_args,
    set_GraphVAE_args,
)
from grade.utils.random_seed import RANDOM_SEED


ROOT = pathlib.Path(__file__).resolve().parents[2]
CANDIDATE_FILES = ('candidates.json', 'candidates_tensor.pt')


def set_beam_search_args(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('beam_search')
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--dataset', choices=['Younger'], default='Younger')
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--max_instance', type=int, default=20)
    parser.add_argument('-ssm', '--subgraph_sampling_method', type=str, default='RESU')
    parser.add_argument('-sd', '--sampling_density', type=int, default=400000)
    parser.add_argument('--selection', choices=['topk', 'stochastic'], default='topk')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0', choices=['cpu', '0', '1', '2', '3'], help="The GPU device to be used.")
    models = parser.add_subparsers(required=True)
    model_setters = [
        ('DisCo', set_DisCo_args),
        ('DiGress', set_DiGress_args),
        ('DeFoG', set_DeFoG_args),
        ('GraphVAE', set_GraphVAE_args),
        ('GNNRanker', set_GNNRanker_args),
    ]
    for name, set_model_args in model_setters:
        set_model_args(models)
        model_parser = models.choices[name]
        model_parser.set_defaults(
            score_run=model_parser.get_default('run'),
            run=beam_search_run,
            algo=name,
        )


def candidate_dir(args, size):
    root = ROOT / 'data' / args.dataset
    prefix = f'{size}-node-{args.subgraph_sampling_method}-{args.sampling_density}-{args.algo}'
    if size == 3:
        return root / prefix
    return root / f'{prefix}-{args.topK}-{args.max_instance}-{args.selection}'


def checkpoint_dir(args, size):
    name = f'{args.dataset}_{args.subgraph_sampling_method}_{size}_{args.sampling_density}_{args.algo}'
    return ROOT / 'outputs' / name / 'saved_models'


def require_checkpoint(path):
    # Match the directory checkpoint names accepted by load_checkpoint.
    if not path.is_dir() or not any(
        re.match(r'^(\d+)_([0-9.]+)$', file.stem)
        for file in path.glob('*.pt') if file.is_file()
    ):
        raise FileNotFoundError(f'No usable model checkpoint in: {path}')


def complete(directory, filenames):
    return all((directory / filename).is_file() for filename in filenames)


def beam_search_run(args):
    args.seed = RANDOM_SEED
    if args.k < 4:
        raise ValueError('k must be at least 4')
    if args.topK < 1 or args.max_instance < 1:
        raise ValueError('topK and max_instance must be positive')

    root = ROOT / 'data' / args.dataset
    if not (root / 'all_3.json').is_file():
        print('[beam_search] Enumerating 3-node patterns', flush=True)
        enumerate_subgraphs(3)

    initial_dir = candidate_dir(args, 3)
    if not complete(initial_dir, CANDIDATE_FILES + ('probs.json',)):
        print('[beam_search] Initializing shared 3-node candidates', flush=True)
        initial_args = copy.copy(args)
        initial_args.subgraph_size = 3
        initial_args.work_dir = initial_dir
        initialize(initial_args)

    for size in range(4, args.k + 1):
        work_dir = candidate_dir(args, size)
        if not complete(work_dir, CANDIDATE_FILES):
            print(f'[beam_search] Expanding {size - 1}-node to {size}-node', flush=True)
            expand_args = copy.copy(args)
            expand_args.subgraph_size = size - 1
            expand_args.work_dir = candidate_dir(args, size - 1)
            expand_args.next_work_dir = work_dir
            expand(expand_args)

        probs_path = work_dir / 'probs.json'
        if probs_path.is_file():
            print(f'[beam_search] Reusing {size}-node scores: {probs_path}', flush=True)
            continue

        output_dir = work_dir
        logps_path = output_dir / 'score_logps.json'
        if not logps_path.is_file():
            model_path = checkpoint_dir(args, size)
            require_checkpoint(model_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f'[beam_search] Scoring {size}-node candidates', flush=True)
            score_args = copy.copy(args)
            score_args.mode = 'score'
            score_args.subgraph_size = size
            score_args.score_dataset_path = str(work_dir / 'candidates_tensor.pt')
            score_args.output_dirpath = str(output_dir)
            score_args.model_load_path = str(model_path)
            score_args.model_save_dirpath = None
            args.score_run(score_args)

        print(f'[beam_search] Aggregating {size}-node scores', flush=True)
        aggregate(argparse.Namespace(
            score_logps_path=logps_path,
            probs_path=probs_path,
        ))
