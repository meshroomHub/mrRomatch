import math


def canonicalize(pairs, order):
    """
    Merge directed pairs into canonical (earlier, later) pairs.

    Args:
        pairs (list[tuple[int, int]]): Directed view id pairs.
        order (dict[int, int]): View id -> frame rank.

    Returns:
        dict[tuple[int, int], set[str]]: Key (a, b) with order[a] < order[b].
            Value is a subset of {"AB", "BA"}: "AB" if (a, b) was listed,
            "BA" if (b, a) was listed. Self-pairs are dropped.
    """
    canon = {}
    for a, b in pairs:
        if a == b:
            continue
        if order[a] < order[b]:
            canon.setdefault((a, b), set()).add("AB")
        else:
            canon.setdefault((b, a), set()).add("BA")
    return canon


def assign_chunks(canon, order, chunk_count):
    """
    Split canonical pairs into chunks of near-equal size that touch few views.

    Pairs are grouped into square tiles over the frame-rank plane, and tiles are
    packed row-major into chunks of at most ceil(len(canon) / chunk_count) pairs
    (a tile may be split across chunks). Deterministic.

    Args:
        canon (dict[tuple[int, int], set[str]]): Output of canonicalize().
        order (dict[int, int]): View id -> frame rank.
        chunk_count (int): Number of chunks to return.

    Returns:
        list[list[tuple[int, int]]]: chunk_count lists of canonical pairs.
    """
    per_chunk = max(1, math.ceil(len(canon) / chunk_count))
    side = max(1, math.ceil(math.sqrt(per_chunk)))

    tiles = {}
    for a, b in canon:
        tiles.setdefault((order[a] // side, order[b] // side), []).append((a, b))

    chunks = [[]]
    for key in sorted(tiles):
        for pair in sorted(tiles[key], key=lambda p: (order[p[0]], order[p[1]])):
            if len(chunks[-1]) >= per_chunk:
                chunks.append([])
            chunks[-1].append(pair)

    chunks += [[] for _ in range(chunk_count - len(chunks))]
    return chunks


def chunk_views(pairs):
    """
    Args:
        pairs (list[tuple[int, int]]): Pairs of view ids.

    Returns:
        set[int]: All view ids used by the pairs.
    """
    return {v for pair in pairs for v in pair}


def order_for_cache(pairs, order, capacity):
    """
    Reorder pairs so consecutive pairs reuse the same views (LRU cache friendly).

    Args:
        pairs (list[tuple[int, int]]): Pairs of view ids.
        order (dict[int, int]): View id -> frame rank.
        capacity (int): Number of views the cache can hold.

    Returns:
        list[tuple[int, int]]: The pairs sorted by blocks of capacity // 2 ranks.
    """
    block = max(1, capacity // 2)
    return sorted(pairs, key=lambda p: (order[p[0]] // block, order[p[1]] // block, order[p[0]], order[p[1]]))
