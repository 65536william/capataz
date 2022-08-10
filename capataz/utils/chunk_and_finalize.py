from .pad_or_pack_docs import pad_or_pack_docs
from .enforce_min_unique import enforce_min_unique
import random

def chunk_and_finalize(tokenized_docs, args, tokenizer):

    sequences = pad_or_pack_docs(tokenized_docs, args["context_size"], tokenizer)

    if args["min_unique_tokens"] > 0:
        sequences = list(
            enforce_min_unique(
                sequences, args["min_unique_tokens"]
            )
        )

    if args["shuffle"]:
        random.shuffle(full_seqs)
    
    return sequences
