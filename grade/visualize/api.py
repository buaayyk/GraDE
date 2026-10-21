import argparse

from grade.utils.log_parser import parse_training_log
from grade.visualize.visualize import plot_training_loss, plot_X_acc, plot_E_acc

def visualize_run(args: argparse.Namespace):
    logs = parse_training_log(args.log_path)
    if args.metric == 'loss':
        plot_training_loss(logs, args.fig_path)
    elif args.metric == 'X_acc':
        plot_X_acc(logs, args.fig_path)
    elif args.metric == 'E_acc':
        plot_E_acc(logs, args.fig_path)

def set_visualize_args(subparsers: argparse._SubParsersAction):
    visualize_parser: argparse.ArgumentParser = subparsers.add_parser('visualize')
    visualize_parser.add_argument('--log_path', type=str, required=True)
    visualize_parser.add_argument('--metric', type=str, choices=['loss', 'X_acc', 'E_acc'], default='loss')
    visualize_parser.add_argument('--fig_path', type=str, required=True)
    visualize_parser.set_defaults(run=visualize_run)