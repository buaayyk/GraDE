import argparse
from grade.datasets.api import set_datasets_args
from grade.diffusion.api import set_diffusion_args
from grade.evaluation.api import set_evaluation_args
from grade.visualize.api import set_visualize_args
from grade.beam_search.api import set_beam_search_args

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    set_datasets_args(subparsers)
    set_diffusion_args(subparsers)
    set_evaluation_args(subparsers)
    set_visualize_args(subparsers)
    set_beam_search_args(subparsers)
    args = parser.parse_args()
    if hasattr(args, 'run'):
        args.run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()