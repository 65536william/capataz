def enforce_min_unique(seqs, min_unique_tokens, verbose=False):
    for seq in seqs:
        if len(set(seq)) >= min_unique_tokens:
            yield seq