import argparse
import itertools
import os
import re
import random

from pathlib import Path
from unittest import result

from lm_dataformat import Reader
import ftfy
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import torch
import tensorflow as tf
from typing import List

from rich.progress import Progress

import functools
from multiprocessing import Pool

from save_sequences import save_sequences
from utils import split_list, get_file_paths

# adapted from mesh-transformer-jax's create-finetune-tfrecords.py script

# there are files which contain documents


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Converts a text dataset into the training data format expected by the model.
    Adapted from the script create_tfrecords.py in the gpt-neo repo.
    - Your text dataset:
        - can be provided as .txt files, or as an archive (.tar.gz, .xz, jsonl.zst).
        - can be one file or multiple
            - using a single large file may use too much memory and crash - if this occurs, split the file up into a few files
        - the model's end-of-text separator is added between the contents of each file
        - if the string '<|endoftext|>' appears inside a file, it is treated as the model's end-of-text separator (not the actual string '<|endoftext|>')
            - this behavior can be disabled with --treat-eot-as-text
    This script creates a single .tfrecords file as output
        - Why: the model's data loader ignores "trailing" data (< 1 batch) at the end of a .tfrecords file
            - this causes data loss if you have many .tfrecords files
        - This is probably not appropriate for very large datasets
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an input file, or a directory that contains the input files.",
    )
    parser.add_argument(
        "output_name",
        type=str,
        help="Name of output file will be {name}_{seqnum}.tfrecords, where seqnum is total sequence count",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default="../",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt-2",
        help="Tokenizer",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="pt",
        help="Output format. Defaults to .pt",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=2048,
        help="Training sequence length in tokens. Actual chunk size will be context size + 1 because model labels are shifted one to the right.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of CPU processes",
    )
    parser.add_argument(
        "--groups-per-file",
        type=int,
        default=16384,
        help="How many groups of sequences will be saved into a single file.",
    )

    cleaning_args = parser.add_argument_group("data cleaning arguments")

    cleaning_args.add_argument(
        "--normalize-with-ftfy", action="store_true", help="Normalize text with ftfy"
    )
    cleaning_args.add_argument(
        "--normalize-with-wikitext-detokenize",
        action="store_true",
        help="Use wikitext detokenizer",
    )
    minu_help = "Exclude repetitive documents made up of < MIN_UNIQUE_TOKENS unique tokens. These can produce large gradients."
    cleaning_args.add_argument(
        "--min-unique-tokens", type=int, default=192, help=minu_help
    )

    shuffle_pack_args = parser.add_argument_group("data shuffling/packing arguments")
    repack_ep_help = "Repeat the data num-repacks times, shuffled differently in each repetition. Recommended for multi-epoch training (set this to your intended number of epochs)."
    shuffle_pack_args.add_argument(
        "--num-repacks", type=int, default=1, help=repack_ep_help
    )
    shuffle_pack_args.add_argument(
        "--shuffling-seed",
        type=int,
        default="-1",
        help="random seed for shuffling data",
    )
    shuffle_pack_args.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Disables shuffling, so the input and output data have the same order.",
    )

    misc_args = parser.add_argument_group("miscellaneous arguments")
    misc_args.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Prints extra information, such as the text removed by --min-unique-tokens",
    )

    args = parser.parse_args()

    # convert input_path to pathy
    args.input_path = Path(args.input_path)

    return args

def capataz(args):
    Path.mkdir(Path(args["output_directory"]), parents=True, exist_ok=True)
    file_paths = get_file_paths(args["input_path"])
    if len(file_paths) > args["num_threads"]:  # do multiprocessing
        split_file_paths = split_list(file_paths, len(file_paths) // args["num_threads"])
        with Pool(processes=args["num_threads"]) as pool:
            pool.starmap(
                save_sequences,
                [(split_file_paths[i], args) for i in range(args["num_threads"])],
            )
    else:  # single cpu process
        save_sequences(file_paths, args)
    return True


if __name__ == "__main__":
    args = vars(parse_args())
    capataz(args)
