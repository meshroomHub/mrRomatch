from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat
from pyalicevision import system as avsys

from common import *

import math
import os
import time
import logging
import h5py, hdf5plugin

from pathlib import Path


def _acquire_archive_lock(lock_path, timeout_s=600.0, poll_s=0.1, stale_lock_s=3600.0):
    """
    Acquire an inter-process lock file with timeout and stale lock handling.

    Args:
        lock_path (str): Path of the lock file to create.
        timeout_s (float): Maximum time to wait before failing.
        poll_s (float): Sleep duration between retries.
        stale_lock_s (float): If an existing lock is older than this threshold,
            it is considered stale and removed.
    """
    start = time.monotonic()

    while True:
        try:
            lfd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(lfd, "w") as lock_fd:
                lock_fd.write(f"pid={os.getpid()}\n")
                lock_fd.write(f"created={time.time()}\n")
            return
        except FileExistsError:
            # Try to recover from stale lock files left behind by crashes.
            try:
                lock_age = time.time() - os.path.getmtime(lock_path)
                if lock_age > stale_lock_s:
                    logging.warning(f"Removing stale lock file: {lock_path}")
                    os.unlink(lock_path)
                    continue
            except FileNotFoundError:
                # Lost race: lock disappeared between exists check and stat.
                continue

            waited = time.monotonic() - start
            if waited >= timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for lock: {lock_path}"
                )

            time.sleep(poll_s)


def _release_archive_lock(lock_path):
    """Release lock file if it exists."""
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass


def parse_masks_folders(masksFolders):
    """
    Normalise the masks-folder argument into a deduplicated list of path strings.

    Accepts ``None``, a single path string, or a (possibly nested) list/tuple of
    path strings. Empty strings and whitespace-only entries are silently ignored.
    Duplicate paths are removed while preserving order.

    Args:
        masksFolders: Path string, list/tuple of path strings, or ``None``.

    Returns:
        list[str]: Ordered, deduplicated list of non-empty folder paths.
    """
    if masksFolders is None:
        return []

    if isinstance(masksFolders, (list, tuple)):
        folders = []
        for item in masksFolders:
            folders.extend(parse_masks_folders(item))
        deduped = []
        seen = set()
        for folder in folders:
            if folder not in seen:
                deduped.append(folder)
                seen.add(folder)
        return deduped

    if not isinstance(masksFolders, str):
        return []

    raw = masksFolders.strip()
    if len(raw) == 0:
        return []

    return [raw]


def find_mask_paths(masksFolders, mask_filename):
    """
    Search a list of folders for a mask file and return all matches.

    Args:
        masksFolders (list[str]): Ordered list of directories to search.
        mask_filename (str): Filename of the mask (e.g. ``"frame_0001.png"``).

    Returns:
        list[str]: Paths to all existing mask files found across the provided
            folders, in order. Empty list if none are found.
    """
    paths = []
    for folder in masksFolders:
        candidate = os.path.join(folder, mask_filename)
        if os.path.exists(candidate):
            paths.append(candidate)
    return paths


def apply_masks(inputSfMData, imagePairsList, warpArchive, confidenceArchive, masksFolders, masksExtension, outputConfidenceArchive, rangeIteration, rangeBlocksCount):
    """
    Apply binary masks to dense confidence maps and write the results to disk.

    For each image pair in the assigned processing range the function:
    - loads the warp and confidence EXR files,
    - looks up a mask for the reference image and, if found, zeros out confidence
      at all reference pixels covered by the mask,
    - looks up a mask for the other image and, if found, follows the warp to
      determine which reference pixels map to masked regions in the other image
      and zeros their confidence as well,
    - skips the pair entirely if no reference mask is found,
    - writes the filtered confidence map to ``outputConfidenceArchive``.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        warpArchive (str): Directory containing warp EXR files.
        confidenceArchive (str): Directory containing input confidence EXR files.
        masksFolders (str | list[str] | None): One or more directories to search
            for mask images. Parsed by :func:`parse_masks_folders`.
        masksExtension (str): File extension of the mask images (e.g. ``"png"``).
        outputConfidenceArchive (str): Directory where filtered confidence EXR
            files are written.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    pairsToProcessTmp = list(plist)
    pairsToProcess = list()
    for pair in pairsToProcessTmp:
        if not pair[0] in iinfos or not pair[1] in iinfos:
            continue
        pairsToProcess.append(pair)

    
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(pairsToProcess))
    if not valid:
        logging.error("Error computing range.")
        raise RuntimeError("Error computing range.")

    masksFolders = parse_masks_folders(masksFolders)
    if len(masksFolders) == 0:
        logging.error("At least one masks folder is required.")
        raise RuntimeError("At least one masks folder is required.")

    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]
    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")
    

    # loop over pairs of images
    for item in pairsToProcess:
        
        #id of views
        referenceId = item[0]
        otherId = item[1]

        # retrieve ImageInfos
        reference_iinfo = iinfos[referenceId]
        other_iinfo = iinfos[otherId]

        #Build paths
        pair_string = str(referenceId) + "_" + str(otherId)

        logging.info(f"Processing pair {pair_string}")

        #load images
        with h5py.File(warpArchive, "r") as f_warp_h5, \
             h5py.File(confidenceArchive, "r") as f_conf_h5:
            if pair_string not in f_conf_h5 or pair_string not in f_warp_h5:
                continue
            warp_A_B = f_warp_h5[pair_string][()].astype(np.float32)
            confidence_A_B = f_conf_h5[pair_string][()].astype(np.float32) / 255.0

        #get properties
        warpHeight = confidence_A_B.shape[0]
        warpWidth = confidence_A_B.shape[1]

        #upgrade coordinates
        warp_A_B[:, :, 0] *= warp_A_B.shape[1]
        warp_A_B[:, :, 1] *= warp_A_B.shape[0]

        #Build resized masks
        maskReference = None
        maskOther = None
        # Replace the extension with the mask extension
        stem = os.path.splitext(os.path.basename(reference_iinfo.path))[0]
        mask_filename = f"{stem}.{masksExtension}"

        # Build the path to the correct mask
        mask_paths = find_mask_paths(masksFolders, mask_filename)

        #Build mask for reference
        if mask_paths:
            logging.info(f"Found reference mask(s) at {mask_paths}")
            maskLarge = open_image(mask_paths[0], isBW=True, isFloat=False)
            maskSmall = avimage.Image_uchar()
            avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False)
            maskReference = np.squeeze(maskSmall.getNumpyArray())
            for extra_path in mask_paths[1:]:
                maskLargeExtra = open_image(extra_path, isBW=True, isFloat=False)
                maskSmallExtra = avimage.Image_uchar()
                avimage.resampleImage(warpWidth, warpHeight, maskLargeExtra, maskSmallExtra, False)
                maskReference = np.minimum(maskReference, np.squeeze(maskSmallExtra.getNumpyArray()))

        # Replace the extension with the mask extension
        stem = os.path.splitext(os.path.basename(other_iinfo.path))[0]
        mask_filename = f"{stem}.{masksExtension}"

        # Build the path to the correct mask
        mask_paths = find_mask_paths(masksFolders, mask_filename)

        if mask_paths:
            logging.info(f"Found comparison mask(s) at {mask_paths}")
            maskLarge = open_image(mask_paths[0], isBW=True, isFloat=False)
            maskSmall = avimage.Image_uchar()
            avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False)
            maskOther = np.squeeze(maskSmall.getNumpyArray())
            for extra_path in mask_paths[1:]:
                maskLargeExtra = open_image(extra_path, isBW=True, isFloat=False)
                maskSmallExtra = avimage.Image_uchar()
                avimage.resampleImage(warpWidth, warpHeight, maskLargeExtra, maskSmallExtra, False)
                maskOther = np.minimum(maskOther, np.squeeze(maskSmallExtra.getNumpyArray()))

        #make sure masks exist
        if maskReference is None:
            continue
        
        #masking using reference mask is straightforward
        confidence_A_B[maskReference == 0] = 0

        if maskOther is not None:
            x = warp_A_B[:, :, 0].astype(np.int64)
            y = warp_A_B[:, :, 1].astype(np.int64)

            valid = (x >= 0) & (y >= 0) & (x < maskOther.shape[1]) & (y < maskOther.shape[0])

            masked_other = np.zeros_like(valid, dtype=bool)
            masked_other[valid] = maskOther[y[valid], x[valid]] == 0
            confidence_A_B[masked_other] = 0
         
        
        #write output
        lock_path = f"{outputConfidenceArchive}.lock"
        _acquire_archive_lock(lock_path)
        try:
            with h5py.File(outputConfidenceArchive, "a") as f_out:
                if pair_string in f_out:
                    del f_out[pair_string]
                f_out.create_dataset(pair_string, data=(confidence_A_B * 255.0).astype(np.uint8), dtype=np.uint8, **hdf5plugin.LZ4(), chunks=True)
        finally:
            _release_archive_lock(lock_path)

if __name__ == '__main__':
    import argparse

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpArchive', type=str, help='')
    parser.add_argument('--confidenceArchive', type=str, help='')
    parser.add_argument('--masksFolders', nargs='*', default=None, help='')
    parser.add_argument('--masksExtension', type=str, help='')
    parser.add_argument('--outputConfidenceArchive', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=apply_masks)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpArchive=args.warpArchive,
                    confidenceArchive=args.confidenceArchive,
                    masksFolders=args.masksFolders,
                    masksExtension=args.masksExtension,
                    outputConfidenceArchive=args.outputConfidenceArchive,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
