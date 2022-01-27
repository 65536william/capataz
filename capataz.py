import argparse
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

from itertools import repeat
from multiprocessing import Pool

# adapted from mesh-transformer-jax's create-finetune-tfrecords.py script


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
        "name",
        type=str,
        help="Name of output file will be {name}_{seqnum}.tfrecords, where seqnum is total sequence count",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
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
        "--threads",
        type=int,
        default=1,
        help="Number of CPU processes",
    )
    parser.add_argument(
        "--chunks-per-file",
        type=int,
        default=16384,
        help="How many chunks will be saved into a single file.",
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
        "--seed",
        type=int,
        help="random seed for shuffling data",
    )
    shuffle_pack_args.add_argument(
        "--preserve-data-order",
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


def get_files(input_path):

    supported_filetypes = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]

    if input_path.is_dir():
        subfiles_by_type = [
            list(Path(input_path).glob(f"*{type}")) for type in supported_filetypes
        ]
        files = [
            sub_file for subfile_group in subfiles_by_type for sub_file in subfile_group
        ]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert (
            str(input_path).endswith(f_type) for f_type in supported_filetypes
        ), f"input filetype must be one of: {supported_filetypes}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"no such file or directory: {input_path=}")

    return [str(file) for file in files]


def wikitext_detokenizer(string):

    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def split_list(list, limit):

    # splits list into limit (or len(list) if len(list) < limit) size chunks
    return [list[i : i + limit] for i in range(0, len(list), limit)]


def enforce_min_unique(seqs, min_unique_tokens, enc, verbose=False):

    for seq in tqdm(seqs, mininterval=1, smoothing=0, desc="enforce_min_unique_tokens"):
        if len(set(seq)) >= min_unique_tokens:
            yield seq
        elif verbose:
            text = enc.decode(seq)
            print(f"excluding with {len(set(seq))} unique tokens:\n\n{repr(text)}\n\n")


def split_by_eos_token(raw_docs, eos_token):

    for raw_doc in raw_docs:
        for raw_subdoc in raw_doc.split(eos_token):
            if len(raw_subdoc) > 0:
                yield raw_subdoc


def clean_and_prepare_and_tokenize_generator(
    raw_docs, tokenizer, normalize_with_ftfy, normalize_with_wikitext_detokenize
):

    for raw_doc in tqdm(raw_docs):
        if normalize_with_ftfy:
            raw_doc = ftfy.fix_text(raw_doc, normalization="NFKC")
        if normalize_with_wikitext_detokenize:
            raw_doc = wikitext_detokenizer(raw_doc)
        tokenized_doc = tokenizer.encode(raw_doc) + [tokenizer.eos_token_id]
        yield tokenized_doc


def tokenized_docs_generator(raw_files, tokenizer, args):

    reader = Reader(raw_files)
    raw_docs = reader.stream_data(threaded=False)
    raw_docs = split_by_eos_token(raw_docs, tokenizer.eos_token)
    return clean_and_prepare_and_tokenize_generator(
        raw_docs,
        tokenizer,
        normalize_with_ftfy=args.normalize_with_ftfy,
        normalize_with_wikitext_detokenize=args.normalize_with_wikitext_detokenize,
    )


def tokenize_docs(raw_files, args, tokenizer):

    if len(raw_files) > 1:
        if args.preserve_data_order:
            print("sorting the raw files")
            raw_files = sorted(raw_files)
        else:
            print("shuffling the raw files")
            random.shuffle(raw_files)

    tokenized_docs = []

    for _ in tqdm(raw_files, mininterval=10, smoothing=0, desc="tokenizing"):
        tokenized_docs.extend(tokenized_docs_generator(raw_files, tokenizer, args))

    if not args.preserve_data_order:
        print("shuffling the tokenized docs")
        random.shuffle(tokenized_docs)

    return tokenized_docs


def split_docs_into_sequences(tokenized_docs, context_size):
    chunk_size = context_size + 1

    accumulating_sequence = []
    for tokenized_doc in tokenized_docs:
        accumulating_sequence.extend(tokenized_doc)
        if len(accumulating_sequence) > chunk_size:
            sequences = split_list(accumulating_sequence, 2049)
            yield from sequences[:-1]
            accumulating_sequence = sequences[-1]

    if len(accumulating_sequence) > 0:
        yield accumulating_sequence


def chunk_and_finalize(tokenized_docs, args, tokenizer):

    sequences = list(split_docs_into_sequences(tokenized_docs, args.context_size))

    full_seqs, trailing_data = sequences[:-1], sequences[-1]

    if args.min_unique_tokens > 0:
        full_seqs = list(
            enforce_min_unique(
                full_seqs, args.min_unique_tokens, tokenizer, args.verbose
            )
        )

    if not args.preserve_data_order:
        random.shuffle(full_seqs)

    return full_seqs, trailing_data


def capataz_pt(raw_files, args):

    print("loading tokenizer")
    if args.tokenizer == "gpt-2":
        GPT2TokenizerFast.max_model_input_sizes["gpt2"] = 1e20  # prevents error
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    else:
        raise ValueError(f"tokenizer `{args.tokenizer}` is not supported")

    random.seed(args.seed if args.seed else None)

    # first
    sequences = []
    tokenized_docs = tokenize_docs(raw_files, args, tokenizer)
    capped_sequences, trailing_data = chunk_and_finalize(
        tokenized_docs, args, tokenizer
    )
    sequences.extend(capped_sequences)
    print(f"there are {len(sequences)} sequences")

    # repacks
    for repeat_idx in range(1, args.num_repacks):
        print(f"repacking on iteration {repeat_idx}")
        if not args.preserve_data_order:
            random.shuffle(tokenized_docs)
            capped_sequences, trailing_data = chunk_and_finalize(
                tokenized_docs, args, tokenizer
            )
        else:
            # if we're preserving data order, we can still "repack" by shifting everything
            # with the trailing data of the last epoch at the beginning
            seqs_with_prefix = [trailing_data] + capped_sequences
            capped_sequences, trailing_data = chunk_and_finalize(
                seqs_with_prefix, args, tokenizer
            )

        sequences.extend(capped_sequences)

    # final
    print(f"dropped {len(trailing_data)} trailing tokens")

    sequences = split_list(sequences, args.chunks_per_file)

    for idx, chunk_group in enumerate(sequences):
        total_chunk_len = len(chunk_group)
        new_file_path = os.path.join(
            args.output_dir, f"{args.name}_{idx}_{total_chunk_len}.pt"
        )
        print("writing to drive")
        torch.save(torch.tensor(chunk_group, dtype=torch.float16), new_file_path)
        print(f"{args.name}_{idx}_{total_chunk_len}.pt saved")


if __name__ == "__main__":

    args = parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    raw_files = get_files(args.input_path)

    if args.output_format == "pt":
        if args.threads > 1:
            files = split_list(raw_files, len(raw_files) // args.threads)
            with Pool(processes=args.threads) as pool:
                progress_bar = tqdm(pool.imap(capataz_pt, zip(raw_files, repeat(args), range(len(raw_files)))))
                meta = {"discarded": 0, "processed": 0, "successful": 0}
                for results in progress_bar:
                    progress_bar.update()
                    for k, v in results.items():
                        meta[k] += v
                print(meta)
        else:
            capataz_pt(raw_files, args)
    else:
        raise ValueError(f"output format `{args.output_format}` is not supported")
