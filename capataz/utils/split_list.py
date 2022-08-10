def split_list(list, limit):
    # splits list into limit (or len(list) if len(list) < limit) size chunks
    return [list[i : i + limit] for i in range(0, len(list), limit)]
