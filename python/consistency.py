from common import *
import logging
from pathlib import Path
from itertools import pairwise
import os
import math
import numpy as np
from contextlib import ExitStack
import h5py, hdf5plugin
from shardWriter import ShardWriter

def parse_archives(inputArchives):
    """
    Normalise the archive argument into a deduplicated list of path strings.

    Accepts ``None``, a single path string, or a (possibly nested) list/tuple of
    path strings. Empty strings and whitespace-only entries are silently ignored.
    Duplicate paths are removed while preserving order.

    Args:
        inputArchives: Path string, list/tuple of path strings, or ``None``.

    Returns:
        list[str]: Ordered, deduplicated list of non-empty archive paths.
    """
    if inputArchives is None:
        return []

    if isinstance(inputArchives, (list, tuple)):
        archives = []
        for item in inputArchives:
            archives.extend(parse_archives(item))
        deduped = []
        seen = set()
        for archive in archives:
            if archive not in seen:
                deduped.append(archive)
                seen.add(archive)
        return deduped

    if not isinstance(inputArchives, str):
        return []

    raw = inputArchives.strip()
    if len(raw) == 0:
        return []

    return [raw]


def find_dataset(archives, key):
    """
    Search a list of opened archives for a dataset and return the first match.

    Args:
        archives (list[h5py.File]): Ordered list of opened HDF5 archives.
        key (str): Dataset name (e.g. ``"123_456"``).

    Returns:
        numpy.ndarray | None: Content of the first matching dataset, or ``None``
            if the key is not found in any of the provided archives.
    """
    for archive in archives:
        if key in archive:
            return archive[key][()]
    return None


def read_warp(archives, key):
    """
    Read a warp from the archives as float32, or ``None`` if it is missing.
    """
    warp = find_dataset(archives, key)
    if warp is None:
        return None
    return warp.astype(np.float32)

def check_consistency(warp_A_B, warp_A_C, warp_B_C):
    """
    Compute the triplet-consistency error for three dense warps.

    Given warps A→B, A→C, and B→C, the function checks whether the composition
    B→C ∘ A→B agrees with A→C at every pixel. For each pixel in A it follows
    A→B to a location in B, samples B→C at that location (using the four
    nearest-neighbour corners), and measures the distance between the result
    and the direct A→C prediction.

    Args:
        warp_A_B (numpy.ndarray | None): Dense warp from A to B, shape (H, W, 3)
            with xy coordinates normalised to [0, 1] in the first two channels.
        warp_A_C (numpy.ndarray | None): Dense warp from A to C, same layout.
        warp_B_C (numpy.ndarray | None): Dense warp from B to C, same layout.

    Returns:
        numpy.ndarray | None: Per-pixel consistency error in pixels, shape
            (H, W, 1), or ``None`` if any warp is ``None`` or the shapes differ.
    """
    if warp_A_B is None or warp_A_C is None or warp_B_C is None:
        return None

    if warp_A_B.shape != warp_A_C.shape or warp_A_B.shape != warp_B_C.shape:
        return None

    # Vectorized computation
    height, width = warp_A_B.shape[0], warp_A_B.shape[1]
        
    # Compute normalized coordinates
    x_A_B = warp_A_B[:, :, 0] * width
    y_A_B = warp_A_B[:, :, 1] * height
    x_A_C = warp_A_C[:, :, 0] * width
    y_A_C = warp_A_C[:, :, 1] * height
    
    # Get integer and fractional parts
    ixm = np.floor(x_A_B).astype(int)
    iym = np.floor(y_A_B).astype(int)
    ixp = ixm + 1
    iyp = iym + 1
    
    # Create mask for coordinates inside the image (before clipping)
    # Neighbors past the last row/column are clipped back onto it
    valid = (ixm >= 0) & (iym >= 0) & (ixm < width) & (iym < height)
    
    # Clip indices to valid range to prevent out-of-bounds access
    ixm = np.clip(ixm, 0, width - 1)
    iym = np.clip(iym, 0, height - 1)
    ixp = np.clip(ixp, 0, width - 1)
    iyp = np.clip(iyp, 0, height - 1)
    
    # Sample 4 neighbors
    dx = warp_B_C[iym, ixm, 0] * width - x_A_C
    dy = warp_B_C[iym, ixm, 1] * height - y_A_C
    dist_1 = np.sqrt(dx**2 + dy**2)
    
    dx = warp_B_C[iym, ixp, 0] * width - x_A_C
    dy = warp_B_C[iym, ixp, 1] * height - y_A_C
    dist_2 = np.sqrt(dx**2 + dy**2)
    
    dx = warp_B_C[iyp, ixm, 0] * width - x_A_C
    dy = warp_B_C[iyp, ixm, 1] * height - y_A_C
    dist_3 = np.sqrt(dx**2 + dy**2)
    
    dx = warp_B_C[iyp, ixp, 0] * width - x_A_C
    dy = warp_B_C[iyp, ixp, 1] * height - y_A_C
    dist_4 = np.sqrt(dx**2 + dy**2)
    
    # Take minimum
    dist = np.minimum(np.minimum(dist_1, dist_2), np.minimum(dist_3, dist_4))
    dist = np.where(valid, dist, np.inf)

    return dist[..., np.newaxis]



def compute_consistency(referenceSfMData, framesSfMData, warpArchives, confidenceArchives, outputConfidenceArchive, maxDistance, rangeIteration, rangeBlocksCount):
    """
    Filter confidence maps using triplet warp consistency across a frame sequence.

    For each reference image in the assigned processing range and for each
    consecutive frame pair (A, B) in the frame sequence, the function evaluates
    the consistency of the composition warp(ref→A) ∘ warp(A→B) against
    warp(ref→B) via :func:`check_consistency`. The per-pixel consistency errors
    are accumulated by taking the element-wise minimum across all frame pairs,
    and the confidence map for each ref→frame pair is zeroed out wherever the
    accumulated error exceeds ``maxDistance``.

    Every ref→frame confidence map found in the input archives is written to
    ``outputConfidenceArchive``, filtered when a consistency error is available
    and unchanged otherwise. Each chunk writes its own shard of the output
    archive and links it into the index.

    Args:
        referenceSfMData (str): Path to the SfM data file for the reference views.
        framesSfMData (str): Path to the SfM data file for the frame sequence.
        warpArchives (list[str] | str | None): One or more HDF5 archives to search
            for warps. Parsed by :func:`parse_archives`.
        confidenceArchives (list[str] | str | None): One or more HDF5 archives
            to search for input confidences.
        outputConfidenceArchive (str): HDF5 archive where filtered confidences
            are written (uint8, like RomaMatcher).
        maxDistance (float): Maximum acceptable consistency error in pixels;
            confidence is zeroed above this threshold.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    from pyalicevision import system as avsys

    # Parse sfm
    refinfos = get_imageinfos_from_sfmdata(referenceSfMData)
    framesinfos = get_imageinfos_from_sfmdata(framesSfMData)

    frame_items = list(framesinfos.items())
    ref_items = list(refinfos.items())

    # Computeing parallelization parameters
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(ref_items))
    if not valid:
        logging.error("Range is out of bounds.")
        return
        
    ref_items = ref_items[rangeStart:rangeEnd]

    with ExitStack() as stack:
        warps = [stack.enter_context(h5py.File(path, "r")) for path in parse_archives(warpArchives)]
        confidences = [stack.enter_context(h5py.File(path, "r")) for path in parse_archives(confidenceArchives)]
        writer = ShardWriter(outputConfidenceArchive, rangeIteration)

        try:
            for keyref, _ in ref_items:

                previousDistanceMap = None

                for idxFrame, (k1, _) in enumerate(frame_items):

                    k2 = k1
                    if idxFrame + 1 < len(frame_items):
                        k2, _ = frame_items[idxFrame + 1]

                    # ref -> A, ref -> B, A -> B
                    warp_ref_A = read_warp(warps, str(keyref) + "_" + str(k1))
                    warp_ref_B = read_warp(warps, str(keyref) + "_" + str(k2))
                    warp_A_B = read_warp(warps, str(k1) + "_" + str(k2))

                    distanceMap = check_consistency(warp_ref_A, warp_ref_B, warp_A_B)

                    if previousDistanceMap is not None:
                        if distanceMap is None:
                            distanceMap = previousDistanceMap
                        else:
                            distanceMap = np.minimum(distanceMap, previousDistanceMap)

                    if distanceMap is not None:
                        previousDistanceMap = distanceMap

                    #Read source confidence, stored as uint8
                    pair_string = str(keyref) + "_" + str(k1)
                    confidence_keyref_A = find_dataset(confidences, pair_string)
                    if confidence_keyref_A is None:
                        continue

                    if distanceMap is not None:
                        rejected = (distanceMap > maxDistance).reshape(confidence_keyref_A.shape[:2])
                        confidence_keyref_A[rejected] = 0

                    #write output
                    writer.write(pair_string, confidence_keyref_A, np.uint8)
        except BaseException:
            writer.close()
            raise

        writer.close_and_link()

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--referenceSfMData', type=str, help='')
    parser.add_argument('--framesSfMData', type=str, help='')
    parser.add_argument('--warpArchives', nargs='*', default=None, help='')
    parser.add_argument('--confidenceArchives', nargs='*', default=None, help='')
    parser.add_argument('--outputConfidenceArchive', type=str, help='')
    parser.add_argument('--maxDistance', type=float, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_consistency)


    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(referenceSfMData=args.referenceSfMData,
                framesSfMData=args.framesSfMData,
                warpArchives=args.warpArchives,
                confidenceArchives=args.confidenceArchives,
                outputConfidenceArchive=args.outputConfidenceArchive,
                maxDistance=args.maxDistance,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
