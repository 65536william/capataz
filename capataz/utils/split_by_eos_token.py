
def split_by_eos_token(raw_docs, eos_token):
    subdocs = []
    for raw_doc in raw_docs:
        for raw_subdoc in raw_doc.split(eos_token):
            if len(raw_subdoc) > 0:
                subdocs.append(raw_subdoc)
    return subdocs