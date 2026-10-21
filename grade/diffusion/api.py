import argparse
import os
import pathlib
import importlib
import sys
from torch_geometric.loader import DataLoader
import torch

from grade.datasets.loader.NASBench101 import NASBench101Dataset
from grade.datasets.loader.NASBench201 import NASBench201Dataset
from grade.datasets.loader.NASBench301 import NASBench301Dataset
from grade.datasets.loader.NASBenchNLP import NASBenchNLPDataset
from grade.datasets.loader.Younger import YoungerDataset
from grade.datasets.loader.YoungerPart import YoungerPartDataset
from grade.datasets.loader.PROTEINS import PROTEINSDataset
from grade.datasets.loader.SocioPatternsHighSchool import SocioPatternsHighSchoolDataset
from grade.diffusion.tasks import train, test, generate, estimate
from grade.utils.logger import setup_singleton_logger, get_logger
from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.diffusion.GraphVAE import GraphVAE
from grade.diffusion.GNNRanker import GNNRanker

cur_dir = os.path.dirname(os.path.abspath(__file__))

def set_diffusion_args(subparsers: argparse._SubParsersAction):
    diffusion_parser: argparse.ArgumentParser = subparsers.add_parser('diffusion')
    diffusion_parser.add_argument('--mode', type=str, choices=['train', 'test', 'generate', 'estimate', 'score'], default='train')
    diffusion_parser.add_argument('--output_dirpath', type=str, default=os.path.join(cur_dir, "..", "..", "outputs"))
    diffusion_parser.add_argument('--dataset', type=str, choices=["NAS-Bench-101", "NAS-Bench-201", "NAS-Bench-301", "NAS-Bench-NLP", "Younger", "YoungerPart", "PROTEINS", "SocioPatterns-HighSchool"], default="NAS-Bench-101")
    diffusion_parser.add_argument('-ssm', '--subgraph_sampling_method', type=str, default=None)
    diffusion_parser.add_argument('-ss', '--subgraph_size', type=int, default=None)
    diffusion_parser.add_argument('-sd', '--sampling_density', type=float,  default=None)
    diffusion_parser.add_argument('--score_dataset_path', type=str, default=None)
    diffusion_parser.add_argument('--batch_size', type=int, default=4)
    diffusion_parser.add_argument('--num_workers', type=int, default=4)
    diffusion_parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train.')
    diffusion_parser.add_argument('--device', type=str, default='0', choices=['cpu', '0', '1', '2', '3'], help="The GPU device to be used.")
    diffusion_parser.add_argument('-msd', '--model_save_dirpath', type=str, default=None)
    diffusion_parser.add_argument('-mlp', '--model_load_path', type=str, default=None)
    diffusion_subparsers = diffusion_parser.add_subparsers()
    set_DisCo_args(diffusion_subparsers)
    set_DiGress_args(diffusion_subparsers)
    set_DeFoG_args(diffusion_subparsers)
    set_GraphVAE_args(diffusion_subparsers)
    set_GNNRanker_args(diffusion_subparsers)

def diffusion_action(args, diffusion_alg: DiffusionAlgorithm, model_load_path, model_save_dirpath, datasets):
    train_set, val_set, test_set, estimate_set = datasets[0], datasets[1], datasets[2], datasets[3]
    if args.mode == "train":
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train(diffusion_alg, train_loader, val_loader, args.epochs, model_save_dirpath)
    elif args.mode == "test":
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test(diffusion_alg, test_loader, model_load_path)
    elif args.mode == "generate":
        generate(diffusion_alg)
    elif args.mode == "estimate":
        estimate_loader = DataLoader(estimate_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        estimate(diffusion_alg, estimate_loader, model_load_path, pathlib.Path(args.output_dirpath) / "estimate_logps.json", args.rounds)
    elif args.mode == "score":
        assert args.score_dataset_path is not None
        score_set = torch.load(args.score_dataset_path, weights_only=False)
        score_loader = DataLoader(score_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        estimate(diffusion_alg, score_loader, model_load_path, pathlib.Path(args.output_dirpath) / "score_logps.json", args.rounds)
    pass

def DisCo_run(args: argparse.Namespace):
    if args.dataset in ["PROTEINS", "SocioPatterns-HighSchool"] and getattr(args, "level_fea", 0) != 0:
        args.level_fea = 0
    model_save_dirpath = pathlib.Path(args.output_dirpath) / "saved_models"
    model_load_path = pathlib.Path(args.model_load_path) if args.model_load_path is not None else None
    logger_dir = pathlib.Path(args.output_dirpath)
    setup_singleton_logger(f"DisCo_{args.mode}_{args.dataset}", logger_dir)
    logger = get_logger()
    logger.info(f"args: {args}")

    train_set, val_set, test_set, estimate_set = get_dataset(args)

    disco_dir = os.path.join(os.path.dirname(__file__), 'DisCo')
    sys.path.insert(0, disco_dir)
    try:
        mod = importlib.import_module('disco')
        DisCo = getattr(mod, 'DisCo')
    finally:
        sys.path.remove(disco_dir)

    disco = DisCo(args, train_set, val_set, test_set)

    diffusion_action(
        args, disco, model_load_path, model_save_dirpath, 
        [train_set, val_set, test_set, estimate_set]
    )

def set_DisCo_args(diffusion_subparsers: argparse._SubParsersAction):
    DisCo_parser: argparse.ArgumentParser = diffusion_subparsers.add_parser('DisCo')
    """ ========================== General training settings ========================== """
    DisCo_parser.add_argument('--graph_domain', type=str, default='spectre', choices=['spectre', 'mol'])
    DisCo_parser.add_argument('--backbone', type=str, default='GT', choices=['MPNN', 'GT'])
    DisCo_parser.add_argument('--BAR', type=int, default=1, help='show the progress bar or not.')
    """ ========================== Diffusion settings ========================== """
    DisCo_parser.add_argument('--diff_type', type=str, default='marginal', choices=['uniform', 'marginal'], help='The converged dist is uniform or marginal.')
    DisCo_parser.add_argument('--min_time', type=float, default=0.01, help='The min time sampled for training.')
    DisCo_parser.add_argument('--sampling_steps', type=int, default=100)
    DisCo_parser.add_argument('--n_sample', type=int, default=10)
    DisCo_parser.add_argument('--beta', type=float, default=2., help='beta.')
    DisCo_parser.add_argument('--alpha', type=float, default=.8, help='alpha.')
    DisCo_parser.add_argument('--rounds', type=int, default=20, help='Estimation rounds for log probability')
    """ ========================== Parametric model settings ========================== """
    DisCo_parser.add_argument('--lr', type=float, default=2e-4, 
        help='Learning rate.')
    DisCo_parser.add_argument('--wd', type=float, default=5e-12, 
        help='weight decay.')
    DisCo_parser.add_argument('--enable_scheduler', type=bool, default=True)
    DisCo_parser.add_argument('--dropout', type=float, default=0.1, 
        help='dropout percentage.')
    DisCo_parser.add_argument('--lambda_train', type=float, nargs=2, default=[5.0, 0.0], help='Loss weights [lambda_E, lambda_y].')
    DisCo_parser.add_argument('--n_layers', type=int, default=10, 
        help='number of layers.')
    DisCo_parser.add_argument('--n_dim', type=int, default=256, 
        help='The # of hidden dimensions.')
    DisCo_parser.add_argument('--cycle_fea', type=int, default=0, 
        help='Whether to encode the # of cycle features.')
    DisCo_parser.add_argument('--eigen_fea', type=int, default=0, 
        help='Whether to encode the eigen features.')
    DisCo_parser.add_argument('--rwpe_fea', type=int, default=0, 
        help='Whether to encode the random walk positional encoding.')
    DisCo_parser.add_argument('--global_fea', type=int, default=0, 
        help='Whether to use cycle and eigen global features.')
    DisCo_parser.add_argument('--level_fea', type=int, default=2, 
        help='Whether to use topological sorting level features for DAG.')
    DisCo_parser.add_argument('--extra_time', type=int, default=0, 
        help='add extra time to y')
    DisCo_parser.set_defaults(run=DisCo_run)

def DiGress_run(args: argparse.Namespace):
    model_save_dirpath = pathlib.Path(args.output_dirpath) / "saved_models"
    model_load_path = pathlib.Path(args.model_load_path) if args.model_load_path is not None else None
    logger_dir = pathlib.Path(args.output_dirpath)
    setup_singleton_logger(f"DiGress_{args.mode}_{args.dataset}", logger_dir)
    logger = get_logger()
    logger.info(f"args: {args}")

    train_set, val_set, test_set, estimate_set = get_dataset(args)

    digress_dir = os.path.join(os.path.dirname(__file__), 'DiGress')
    sys.path.insert(0, digress_dir)
    try:
        mod = importlib.import_module('digress')
        DiGress = getattr(mod, 'DiGress')
    finally:
        sys.path.remove(digress_dir)

    digress = DiGress(args, train_set, val_set, test_set)

    diffusion_action(
        args, digress, model_load_path, model_save_dirpath, 
        [train_set, val_set, test_set, estimate_set]
    )


def set_DiGress_args(diffusion_subparsers: argparse._SubParsersAction):
    DiGress_parser: argparse.ArgumentParser = diffusion_subparsers.add_parser('DiGress')
    """ ============ Diffusion setting ============ """
    DiGress_parser.add_argument('--diff_type', type=str, default='marginal', choices=['uniform', 'marginal'], help='The converged dist is uniform or marginal.')
    DiGress_parser.add_argument('--sampling_steps', type=int, default=500)
    DiGress_parser.add_argument('--diffusion_noise_schedule', type=str, choices=['cosine', 'custom'], default='cosine')
    DiGress_parser.add_argument('--n_sample', type=int, default=10)
    DiGress_parser.add_argument('--p_uncondition', type=float, default=1, 
        help='P(train unconditionally)')
    DiGress_parser.add_argument('--gamma', type=float, default=2, 
        help='guidance scale')
    DiGress_parser.add_argument('--conditional', type=int, default=0, 
        help='generate/estimate conditionally or not')
    DiGress_parser.add_argument('--rounds', type=int, default=20, help='Estimation rounds for log probability')
    DiGress_parser.add_argument(
        '--use_full_y',
        type=int,
        choices=[0, 1],
        default=1,
        help='Whether to feed the full y=[y_t, extra_y] into the model.',
    )
    DiGress_parser.add_argument(
        '--use_sin_pe',
        type=int,
        choices=[0, 1],
        default=0,
        help='Whether to add SinPE to node representations.',
    )

    """ ============ Parametric model settings ============="""
    DiGress_parser.add_argument('--lr', type=float, default=2e-4, 
        help='Learning rate.')
    DiGress_parser.add_argument('--wd', type=float, default=1e-12, help='weight decay')
    DiGress_parser.add_argument('--lambda_train', type=float, nargs=2, default=[5.0, 0.0], help='Loss weights [lambda_E, lambda_y].')
    DiGress_parser.add_argument('--n_layers', type=int, default=5)
    DiGress_parser.set_defaults(run=DiGress_run)

def DeFoG_run(args: argparse.Namespace):
    model_save_dirpath = pathlib.Path(args.output_dirpath) / "saved_models"
    model_load_path = pathlib.Path(args.model_load_path) if args.model_load_path is not None else None
    logger_dir = pathlib.Path(args.output_dirpath)
    setup_singleton_logger(f"DeFoG_{args.mode}_{args.dataset}", logger_dir)
    logger = get_logger()
    logger.info(f"args: {args}")

    train_set, val_set, test_set, estimate_set = get_dataset(args)

    defog_dir = os.path.join(os.path.dirname(__file__), 'DeFoG')
    sys.path.insert(0, defog_dir)
    try:
        mod = importlib.import_module('defog')
        DeFoG = getattr(mod, 'DeFoG')
    finally:
        sys.path.remove(defog_dir)

    defog = DeFoG(args, train_set, val_set, test_set)

    diffusion_action(
        args, defog, model_load_path, model_save_dirpath, 
        [train_set, val_set, test_set, estimate_set]
    )


def set_DeFoG_args(diffusion_subparsers: argparse._SubParsersAction):
    DeFoG_parser: argparse.ArgumentParser = diffusion_subparsers.add_parser('DeFoG')
    """ ========================== Diffusion settings ========================== """
    DeFoG_parser.add_argument('--diff_type', type=str, choices=["marginal", "edge_marginal", "node_marginal", "absorbing", "argmax", "absorbfirst", "uniform"], default="marginal")
    DeFoG_parser.add_argument('--sampling_steps', type=int, default=1000)
    DeFoG_parser.add_argument('--n_sample', type=int, default=10)
    DeFoG_parser.add_argument('--conditional', type=int, default=0)
    DeFoG_parser.add_argument('--rdb', type=str, choices=["general", "column",  "entry"], default='general')
    DeFoG_parser.add_argument('--rdb_crit', type=str, default='dummy')
    DeFoG_parser.add_argument('--eta', type=float, default=0)
    DeFoG_parser.add_argument('--omega', type=float, default=0)
    DeFoG_parser.add_argument('--rounds', type=int, default=20)
    DeFoG_parser.add_argument('--level_fea', type=int, default=2)
    """ ============ Parametric model settings ============="""
    DeFoG_parser.add_argument('--lr', type=float, default=2e-4, 
        help='Learning rate.')
    DeFoG_parser.add_argument('--wd', type=float, default=1e-12, help='weight decay')
    DeFoG_parser.add_argument('--lambda_train', type=float, nargs=2, default=[5.0, 0.0], help='Loss weights [lambda_E, lambda_y].')
    DeFoG_parser.add_argument('--n_layers', type=int, default=5)
    DeFoG_parser.add_argument('--extra_fea', type=str, choices=['all', 'cycles', 'eigenvalues', 'rrwp', 'rrwp_comp', None], default=None, help='Add extra features or not')
    DeFoG_parser.add_argument('--rrwp_steps', type=int, default=12)
    """ ============ Time Distortion ======================"""
    DeFoG_parser.add_argument('--train_time_distortion', choices=['identity', 'cos', 'polyinc', 'polydec'], default='identity')
    DeFoG_parser.add_argument('--sample_time_distortion', choices=['identity', 'cos', 'polyinc', 'polydec'], default='identity')
    DeFoG_parser.set_defaults(run=DeFoG_run)


def GraphVAE_run(args: argparse.Namespace):
    model_save_dirpath = pathlib.Path(args.output_dirpath) / "saved_models"
    model_load_path = pathlib.Path(args.model_load_path) if args.model_load_path is not None else None
    logger_dir = pathlib.Path(args.output_dirpath)
    setup_singleton_logger(f"GraphVAE_{args.mode}_{args.dataset}", logger_dir)
    logger = get_logger()
    logger.info(f"args: {args}")

    train_set, val_set, test_set, estimate_set = get_dataset(args)

    graph_vae = GraphVAE(args, train_set, val_set, test_set)

    diffusion_action(
        args, graph_vae, model_load_path, model_save_dirpath,
        [train_set, val_set, test_set, estimate_set]
    )

def set_GraphVAE_args(diffusion_subparsers: argparse._SubParsersAction):
    GraphVAE_parser: argparse.ArgumentParser = diffusion_subparsers.add_parser('GraphVAE')
    GraphVAE_parser.add_argument('--hidden_dim', type=int, default=256)
    GraphVAE_parser.add_argument('--decoder_hidden_dim', type=int, default=256)
    GraphVAE_parser.add_argument('--encoder_layers', type=int, default=3)
    GraphVAE_parser.add_argument('--latent_dim', type=int, default=128)
    GraphVAE_parser.add_argument('--dropout', type=float, default=0.1)
    GraphVAE_parser.add_argument('--kl_beta', type=float, default=1e-3)
    GraphVAE_parser.add_argument('--lr', type=float, default=1e-3)
    GraphVAE_parser.add_argument('--wd', type=float, default=0.0)
    GraphVAE_parser.add_argument('--n_sample', type=int, default=10)
    GraphVAE_parser.add_argument('--enable_scheduler', action='store_true', default=False)
    GraphVAE_parser.add_argument('--rounds', type=int, default=20)
    GraphVAE_parser.set_defaults(run=GraphVAE_run)


def GNNRanker_run(args: argparse.Namespace):
    from grade.datasets.subgraph_sampling.frequency_dataset import build_frequency_datasets

    if args.subgraph_sampling_method is None:
        raise ValueError("GNNRanker requires sampled subgraphs via --subgraph_sampling_method.")
    model_save_dirpath = pathlib.Path(args.output_dirpath) / "saved_models"
    model_load_path = pathlib.Path(args.model_load_path) if args.model_load_path is not None else None
    logger_dir = pathlib.Path(args.output_dirpath)
    setup_singleton_logger(f"GNNRanker_{args.mode}_{args.dataset}", logger_dir)
    logger = get_logger()
    logger.info(f"args: {args}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_train_set, _, _, estimate_set = get_dataset(args)
    train_set, val_set, test_set, metadata = build_frequency_datasets(
        base_train_set, seed=args.split_seed
    )
    logger.info(f"Frequency dataset: {metadata}")
    ranker = GNNRanker(args, base_train_set.get_dataset_info())

    diffusion_action(
        args,
        ranker,
        model_load_path,
        model_save_dirpath,
        [train_set, val_set, test_set, estimate_set],
    )


def set_GNNRanker_args(diffusion_subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = diffusion_subparsers.add_parser("GNNRanker")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--rank_loss_weight", type=float, default=1.0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_scheduler", action="store_true", default=False)
    parser.add_argument("--rounds", type=int, default=1)
    parser.set_defaults(run=GNNRanker_run)

def get_dataset(args):
    if args.dataset == "NAS-Bench-101":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBench101"
        train_set = NASBench101Dataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = NASBench101Dataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = NASBench101Dataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBench101Dataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-201":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBench201"
        train_set = NASBench201Dataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = NASBench201Dataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = NASBench201Dataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBench201Dataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-301":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBench301"
        train_set = NASBench301Dataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = NASBench301Dataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = NASBench301Dataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBench301Dataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "NAS-Bench-NLP":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "NASBenchNLP"
        train_set = NASBenchNLPDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = NASBenchNLPDataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = NASBenchNLPDataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = NASBenchNLPDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "Younger":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "Younger"
        train_set = YoungerDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = YoungerDataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = YoungerDataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = YoungerDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "YoungerPart":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "YoungerPart"
        train_set = YoungerPartDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = YoungerPartDataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = YoungerPartDataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = YoungerPartDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "PROTEINS":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "PROTEINS"
        train_set = PROTEINSDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = PROTEINSDataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = PROTEINSDataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = PROTEINSDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    elif args.dataset == "SocioPatterns-HighSchool":
        dataset_root = pathlib.Path(cur_dir) / ".." / ".." / "data" / "SocioPatternsHighSchool"
        train_set = SocioPatternsHighSchoolDataset(dataset_root, 'train', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        val_set = SocioPatternsHighSchoolDataset(dataset_root, 'valid', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        test_set = SocioPatternsHighSchoolDataset(dataset_root, 'test', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
        estimate_set = SocioPatternsHighSchoolDataset(dataset_root, 'estimate', args.subgraph_sampling_method, args.subgraph_size, args.sampling_density)
    else:
        raise NotImplementedError
    return train_set, val_set, test_set, estimate_set
