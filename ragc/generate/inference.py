import argparse
import os

from pprint import pprint
from typing import Mapping, Dict, Literal, Any

from ragc.generate.utils import load_tasks
from ragc.generate.baseline_inference import generate as baseline_generate
from ragc.generate.rag_inference import generate as rag_generate


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-t', '--tasks',
        type=str,
        required=True,
        help='Path to a file with tasks in EvoCodeBench format',
    )
    parser.add_argument(
        '-r', '--repos',
        type=str,
        required=True,
        help='Path to a directory with all repositories. Must be an absolute path.',
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to an output file with completions',
    )
    parser.add_argument(
        '--generate-func',
        type=str,
        choices=['baseline', 'rag'],
        default='rag',
        help="Generation function to be used. Available options: 'baseline' and 'rag'",
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help="Path to .yaml config for inference",
    )

    return parser.parse_args()


def generate_completions(
        tasks: Mapping[str, Dict[str, Any]],
        repos_dir: str | os.PathLike,
        output_path: str | os.PathLike,
        generate_func: Literal['baseline', 'rag'],
        config_path: str | os.PathLike = None,
):
    # make paths absolute if they are not already
    if not os.path.isabs(repos_dir):
        repos_dir = os.path.abspath(repos_dir)
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    # select generation method
    match generate_func:
        case 'baseline':
            _generate_func = baseline_generate
        case 'rag':
            _generate_func = rag_generate
        case default:
            raise ValueError("generate_func shoulde be amongst ['baseline', 'rag']")

    for task in tasks:
        generation = _generate_func(task, repos_dir, config_path=config_path)
        print(generation)


if __name__ == '__main__':
    # parse args
    args = parse_args()
    print('args:')
    pprint(args.__dict__)
    print('-' * 256)

    # load tasks
    print(f'Loading tasks from {args.tasks}...')
    tasks = load_tasks(args.tasks)
    print(f'Loaded {len(tasks)} tasks.')

    # run generation
    generate_completions(
        tasks,
        args.repos,
        args.output,
        generate_func=args.generate_func,
        config_path=args.config,
    )
