from .split_list import split_list
import binpacking


def pad_or_pack_docs(tokenized_docs, context_size, tokenizer):
    pad = False
    pack = True
    trimmed_doc_len = context_size + 1


    if pack:
        smartly_merged_docs = []
        index_lengths = {}
        for index, tokenized_doc in enumerate(tokenized_docs):
            index_lengths[index] = len(tokenized_doc)
        print("packing bins, this is going to take some time")
        bins = binpacking.to_constant_volume(index_lengths, trimmed_doc_len) # use bin packing to minimise data loss
        for new_bin in bins:
            new_sequence = []
            for index in list(new_bin.keys()):
                new_sequence.extend(tokenized_docs[index])
            smartly_merged_docs.append(new_sequence)
    """     else:
        accumulating_sequence = []
        for tokenized_doc in tokenized_docs:
            accumulating_sequence.extend(tokenized_doc)
            if len(accumulating_sequence) > chunk_size:
                sequences = split_list(accumulating_sequence, 2049)
                yield from sequences[:-1]
                accumulating_sequence = sequences[-1]

        if len(accumulating_sequence) > 0:
            yield accumulating_sequence """
        
    resized_docs = []

    # cut away very long ones, this is a shame but later data in the content probably isn't good enough to train on by itself
    for doc in smartly_merged_docs:
        if len(doc) > trimmed_doc_len:
            doc = split_list(doc, trimmed_doc_len)[0]
            resized_docs.append(doc)
        elif len(doc) < trimmed_doc_len:
            doc = doc + [tokenizer.eos_token_id] * (trimmed_doc_len - len(doc)) # use eos_token for padding
            resized_docs.append(doc)
        else:
            resized_docs.append(doc)

    return resized_docs