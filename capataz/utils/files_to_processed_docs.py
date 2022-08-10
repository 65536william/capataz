from lm_dataformat import Reader
from .normalize_doc import normalize_doc
from .split_by_eos_token import split_by_eos_token
from typing import List
from rich.progress import track


import time


def files_to_processed_docs(raw_files: List[str], args, tokenizer):
    processed_docs = []

    for file in raw_files:
        raw_docs = list(open(file, "r").readlines())
        raw_docs = split_by_eos_token(raw_docs, tokenizer.eos_token)
        for raw_doc in track(raw_docs, description="tokenizing", total=len(raw_docs)):
            normalized_doc = normalize_doc(raw_doc)
            tokenized_doc = tokenizer.encode(normalized_doc)
            tokenized_doc = tokenized_doc + [tokenizer.eos_token_id]
            processed_docs.extend([tokenized_doc])

    return processed_docs
