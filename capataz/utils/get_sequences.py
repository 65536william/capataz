from transformers import GPT2TokenizerFast
import random
from .files_to_processed_docs import files_to_processed_docs
from .chunk_and_finalize import chunk_and_finalize
from .split_list import split_list


def get_sequences(raw_files, args):
    """
    raw_files: list of file paths
    """

    print("loading tokenizer")
    if args["tokenizer"] == "gpt-2":
        GPT2TokenizerFast.max_model_input_sizes[
            "gpt2"
        ] = 1e20  # prevents error apparently
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    else:
        raise ValueError(f"tokenizer `{args['tokenizer']}` is not supported")

    random.seed(args["shuffling_seed"])

    # first
    tokenized_docs = files_to_processed_docs(raw_files, args, tokenizer)
    if args["shuffle"]:
        print("shuffling the tokenized docs")
        random.shuffle(tokenized_docs)
    sequences = chunk_and_finalize(
        tokenized_docs, args, tokenizer
    )
    
    print(f"there are {len(sequences)} sequences")

    sequences = split_list(sequences, args["groups_per_file"])
    
    return sequences