"""
RoMa v2 dense matching from the per-image data written by RomaPrepare.

Each unordered pair is computed once. The coarse matcher always runs both
directions; only the listed directions are refined (both when the loop check
is on). Pairs are matched in an order that reuses the per-image fine
features held in a GPU LRU cache. A background loader (:mod:`prefetch`) reads
the views from the prepare archive ahead of the GPU. A single writer thread post-processes the
listed directions and stores them under ``"<ref>_<other>"`` keys.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass

import h5py
import numpy as np
import torch

import romaCore
from directions import finalize_direction
from fineCache import FineCache, capacity_from_vram
from pairTiling import assign_chunks, canonicalize, order_for_cache
from prefetch import Prefetcher
from prepare import load_view
from shardWriter import ShardWriter

ABORT = object()  # writer sentinel: the GPU pass failed, close shards without linking


@dataclass
class Params:
    """
    Matching parameters.

    Attributes:
        min_confidence (float): Pixels below this confidence are zeroed.
        check_loops (bool): Apply the loop-closure filter.
        loop_threshold (float): Loop error threshold in pixels.
        output_covariance (bool): Also write the per-pixel standard deviation.
    """
    min_confidence: float = 0.02
    check_loops: bool = False
    loop_threshold: float = 3.0
    output_covariance: bool = False


def _nbytes(item):
    return sum(t.nbytes for features in item for t in features.values())


def _views_in_order(pairs):
    return [v for pair in pairs for v in pair]


def _refine_directions(listed, params):
    """Directions to refine: the listed ones, or both when the loop check needs the opposite warp."""
    if params.check_loops:
        return ("AB", "BA")
    return tuple(d for d in ("AB", "BA") if d in listed)


def _fine_of(model, view, device):
    return romaCore.fine(model, view["img_lr"].to(device).float(), view["img_hr"].to(device).float())


def _cache_size(model, index, first, device):
    """
    Size the fine-feature cache from free VRAM, measuring one view (``first``, loaded directly).

    Returns:
        tuple: ``(capacity_bytes, images, seed_features)``.
    """
    capacity = capacity_from_vram()
    seed = _fine_of(model, load_view(index, first, "cpu", with_images=True, half=True), device)
    per_view = _nbytes(seed)
    images = max(2, capacity // per_view)
    logging.info(f"Fine cache: {capacity / 2**30:.2f} GiB, {per_view / 2**20:.0f} MiB per view, {images} views")
    return capacity, images, seed


def _make_cache(model, first, seed, capacity, prefetch, device):
    """Build the GPU LRU cache of fine features, seeded with view ``first``; other views come from ``prefetch``."""
    seeds = {first: seed}
    cache = FineCache(capacity, lambda v: seeds.pop(v) if v in seeds else _fine_of(model, prefetch.get(v), device),
                      _nbytes)
    cache.get(first)
    return cache


def _match_pairs(model, pairs, order, index, index_path, dirs, params, device, put):
    """
    Coarse and refine every pair in cache-friendly order.

    Args:
        model: RoMaV2 model.
        pairs (list[tuple[int, int]]): Canonical pairs, non-empty.
        order (dict[int, int]): View id -> rank.
        index (h5py.File): Open prepare archive.
        index_path (str): Its path, for the background loader.
        dirs (dict): Listed directions per canonical pair.
        params (Params): Matching parameters.
        device: Torch device.
        put (callable): Receives ``(a, b, preds)`` for each refined pair.
    """
    first = pairs[0][0]
    capacity, images, seed = _cache_size(model, index, first, device)
    sequence = order_for_cache(pairs, order, images)
    prefetch = Prefetcher(index_path, [first] + _views_in_order(sequence), with_images=True)
    try:
        cache = _make_cache(model, first, seed, capacity, prefetch, device)
        for a, b in sequence:
            fA = [prefetch.get(a)["dino"][level].to(device).float() for level in range(2)]
            fB = [prefetch.get(b)["dino"][level].to(device).float() for level in range(2)]
            c = romaCore.coarse(model, fA, fB)
            directions = _refine_directions(dirs[(a, b)], params)
            put((a, b, romaCore.finalize(romaCore.refine(model, c, cache.get(a), cache.get(b), directions))))
        logging.info(f"Fine cache misses: {cache.misses}")
    finally:
        prefetch.close()


def _write_pair(writers, a, b, dirs, preds, params):
    """
    Post-process and store the listed directions of a refined pair.

    Args:
        writers (dict[str, ShardWriter]): Writers for ``warp``, ``confidence`` and optionally ``covariance``.
        a (int): First view of the canonical pair.
        b (int): Second view of the canonical pair.
        dirs (set[str]): Listed directions, subset of {"AB", "BA"}.
        preds (dict): Output of :func:`romaCore.finalize`.
        params (Params): Matching parameters.
    """
    for d, o, key in (("AB", "BA", f"{a}_{b}"), ("BA", "AB", f"{b}_{a}")):
        if d not in dirs:
            continue
        # the opposite warp is only read by the loop check, which always refines both directions
        opposite = preds.get(f"warp_{o}", preds[f"warp_{d}"])
        warp, conf = finalize_direction(preds[f"warp_{d}"][0], opposite[0], preds[f"overlap_{d}"][0],
                                        params.check_loops, params.loop_threshold, params.min_confidence)
        writers["warp"].write(key, warp, np.float16)
        writers["confidence"].write(key, conf, np.uint8)
        if "covariance" in writers:
            std = (torch.linalg.det(preds[f"precision_{d}"][0]) ** (-1 / 4)).numpy()
            writers["covariance"].write(key, std, np.float16)


def _writer_loop(q, out, chunk, dirs, params, errors):
    """
    Writer thread body: the only code that touches the ShardWriters.

    Consumes ``(a, b, preds)`` items until ``None`` (link the shards into the
    indexes) or ``ABORT`` (close the shards without linking). On error, records
    the exception in ``errors`` and drains the queue so the producer never blocks.
    """
    done = False
    try:
        names = ["warp", "confidence"] + (["covariance"] if params.output_covariance and out.get("covariance") else [])
        writers = {name: ShardWriter(out[name], chunk) for name in names}
        while True:
            item = q.get()
            if item is None or item is ABORT:
                done = True
                break
            a, b, preds = item
            _write_pair(writers, a, b, dirs[(a, b)], preds, params)
        for writer in writers.values():
            if item is ABORT:
                writer.close()
            else:
                writer.close_and_link()
    except BaseException as e:
        errors.append(e)
        while not done and q.get() not in (None, ABORT):
            pass


def run_chunk(model, pairs_canon, dirs, order, prepare_index, out, chunk, params):
    """
    Match the canonical pairs of one chunk and write the listed directions.

    Args:
        model: RoMaV2 model (unused if ``pairs_canon`` is empty).
        pairs_canon (list[tuple[int, int]]): Canonical pairs (a, b), order[a] < order[b].
        dirs (dict[tuple[int, int], set[str]]): Listed directions per pair ("AB" -> key "a_b", "BA" -> key "b_a").
        order (dict[int, int]): View id -> unique rank.
        prepare_index (str): Path of the RomaPrepare archive.
        out (dict[str, str | None]): Index paths ``warp``, ``confidence`` and ``covariance`` (or None).
        chunk (int): Chunk number, selecting the shard files.
        params (Params): Matching parameters.
    """
    from romav2.device import device

    q = queue.Queue(maxsize=8)
    errors = []
    writer = threading.Thread(target=_writer_loop, args=(q, out, chunk, dirs, params, errors), daemon=True)
    writer.start()

    def put(item):
        if errors:
            raise RuntimeError("Writer thread failed")
        q.put(item)

    start = time.perf_counter()
    try:
        with h5py.File(prepare_index, "r") as index:
            if pairs_canon:
                _match_pairs(model, pairs_canon, order, index, prepare_index, dirs, params, device, put)
    except BaseException as e:
        q.put(ABORT)
        writer.join()
        if errors:
            raise errors[0] from e
        raise
    q.put(None)
    writer.join()
    if errors:
        raise errors[0]
    elapsed = time.perf_counter() - start
    if pairs_canon:
        logging.info(f"Matched {len(pairs_canon)} unordered pairs in {elapsed:.1f} s, "
                     f"{1000 * elapsed / len(pairs_canon):.0f} ms per unordered pair")


def plan_chunk(pairs, order, prepared, chunk, chunk_count):
    """
    Drop pairs with unknown views, then select the canonical pairs of one chunk.

    Args:
        pairs (list[tuple[int, int]]): Directed pairs from the pair list.
        order (dict[int, int]): View id -> rank, for the views of the sfmData.
        prepared (set[str]): View ids (as strings) present in the prepare archive.
        chunk (int): Chunk number.
        chunk_count (int): Number of chunks.

    Returns:
        tuple[list, dict]: The chunk's canonical pairs and their listed directions.
    """
    missing = set()
    for view_id in sorted({v for pair in pairs for v in pair}):
        if view_id not in order:
            logging.warning(f"Dropping pairs with view {view_id}: not in the sfmData")
            missing.add(view_id)
        elif str(view_id) not in prepared:
            logging.warning(f"Dropping pairs with view {view_id}: not in the prepare archive")
            missing.add(view_id)
    canon = canonicalize([p for p in pairs if p[0] not in missing and p[1] not in missing], order)
    chunk_pairs = assign_chunks(canon, order, chunk_count)[chunk]
    return chunk_pairs, {p: canon[p] for p in chunk_pairs}


def compute_matches(inputSfMData, imagePairsList, minConfidence, outputWarpArchive, outputConfidenceArchive,
                    outputCovarianceArchive, checkLoops, loopThreshold, outputCovarianceFlag, prepareArchive,
                    rangeIteration, rangeBlocksCount):
    """
    Run RoMa dense matching on one parallelization block of the pair list.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        minConfidence (float): Matches below this confidence are zeroed.
        outputWarpArchive (str): Archive path for warp.
        outputConfidenceArchive (str): Archive path for confidence.
        outputCovarianceArchive (str): Archive path for covariance.
        checkLoops (bool): Apply loop-closure filtering.
        loopThreshold (float): Maximum round-trip error in pixels.
        outputCovarianceFlag (bool): Also write the covariance archive.
        prepareArchive (str): Archive written by RomaPrepare.
        rangeIteration (int): Index of the current processing block.
        rangeBlocksCount (int): Total number of processing blocks.
    """
    from common import avmic, get_imageinfos_from_sfmdata

    infos = get_imageinfos_from_sfmdata(inputSfMData)
    order = {view_id: rank for rank, view_id in enumerate(infos)}

    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    pairs = [(int(p[0]), int(p[1])) for p in plist]
    logging.info(f"Loaded {len(pairs)} pairs from list")

    with h5py.File(prepareArchive, "r") as index:
        prepared = set(index.keys())
    pairs_canon, dirs = plan_chunk(pairs, order, prepared, rangeIteration, rangeBlocksCount)
    logging.info(f"Processing {len(pairs_canon)} unordered pairs in block {rangeIteration}/{rangeBlocksCount}")

    model = None
    if pairs_canon:
        logging.info("Loading model")
        model = romaCore.load_model()
        compiled = romaCore.enable_compile(model)
        logging.info("Refiners compiled" if compiled else "Refiners not compiled: GPU older than Ampere")
    params = Params(min_confidence=minConfidence, check_loops=checkLoops, loop_threshold=loopThreshold,
                    output_covariance=outputCovarianceFlag)
    out = {
        "warp": outputWarpArchive,
        "confidence": outputConfidenceArchive,
        "covariance": outputCovarianceArchive if outputCovarianceFlag else None,
    }
    run_chunk(model, pairs_canon, dirs, order, prepareArchive, out, rangeIteration, params)


if __name__ == '__main__':
    import argparse

    from common import str_to_bool

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(prog='romaMatcher')
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--minConfidence', type=float, help='', default=0.0)
    parser.add_argument('--outputWarpArchive', type=str, help='')
    parser.add_argument('--outputConfidenceArchive', type=str, help='')
    parser.add_argument('--outputCovarianceArchive', type=str, help='')
    parser.add_argument('--checkLoops', type=str_to_bool, help='', default=False)
    parser.add_argument('--loopThreshold', type=float, help='Maximum accepted round-trip distance in pixels', default=3.0)
    parser.add_argument('--outputCovarianceFlag', type=str_to_bool, help='', default=False)
    parser.add_argument('--prepareArchive', type=str, help='Archive written by RomaPrepare')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    args = parser.parse_args()

    compute_matches(**vars(args))
