from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat
from pyalicevision import system as avsys

from common import *

import os
import logging
import h5py, hdf5plugin

from shardWriter import ShardWriter


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


def load_mask(masksFolders, masksExtension, iinfo, width, height):
    """
    Load the mask(s) of an image, resampled to the warp resolution.

    Several masks found across folders are combined with a minimum.

    Args:
        masksFolders (list[str]): Ordered list of directories to search.
        masksExtension (str): Mask file extension.
        iinfo: ImageInfo of the view (its path gives the mask stem).
        width (int): Output width.
        height (int): Output height.

    Returns:
        numpy.ndarray | None: uint8 mask of shape (height, width), or None if
            no mask file exists for this image.
    """
    stem = os.path.splitext(os.path.basename(iinfo.path))[0]
    mask_paths = find_mask_paths(masksFolders, f"{stem}.{masksExtension}")
    if not mask_paths:
        return None

    logging.info(f"Found mask(s) at {mask_paths}")
    mask = None
    for path in mask_paths:
        maskLarge = open_image(path, isBW=True, isFloat=False)
        maskSmall = avimage.Image_uchar()
        avimage.resampleImage(width, height, maskLarge, maskSmall, False)
        current = np.squeeze(maskSmall.getNumpyArray()).copy()
        mask = current if mask is None else np.minimum(mask, current)
    return mask


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
    

    # Resampled masks by (viewId, width, height); None when the view has no mask.
    mask_cache = {}

    def get_mask(viewId, width, height):
        key = (viewId, width, height)
        if key not in mask_cache:
            mask_cache[key] = load_mask(masksFolders, masksExtension, iinfos[viewId], width, height)
        return mask_cache[key]

    with h5py.File(warpArchive, "r") as f_warp_h5, \
         h5py.File(confidenceArchive, "r") as f_conf_h5:
        writer = ShardWriter(outputConfidenceArchive, rangeIteration)
        try:
            # loop over pairs of images
            for referenceId, otherId in pairsToProcess:
                pair_string = str(referenceId) + "_" + str(otherId)

                if pair_string not in f_conf_h5 or pair_string not in f_warp_h5:
                    continue

                conf_ds = f_conf_h5[pair_string]
                warpHeight, warpWidth = conf_ds.shape[0], conf_ds.shape[1]

                # Pairs without a reference mask are not written.
                maskReference = get_mask(referenceId, warpWidth, warpHeight)
                if maskReference is None:
                    continue

                logging.info(f"Processing pair {pair_string}")

                confidence_A_B = conf_ds[()].astype(np.float32) / 255.0

                #masking using reference mask is straightforward
                confidence_A_B[maskReference == 0] = 0

                maskOther = get_mask(otherId, warpWidth, warpHeight)
                if maskOther is not None:
                    warp_A_B = f_warp_h5[pair_string][()].astype(np.float32)

                    #upgrade coordinates
                    warp_A_B[:, :, 0] *= warp_A_B.shape[1]
                    warp_A_B[:, :, 1] *= warp_A_B.shape[0]

                    x = warp_A_B[:, :, 0].astype(np.int64)
                    y = warp_A_B[:, :, 1].astype(np.int64)

                    valid = (x >= 0) & (y >= 0) & (x < maskOther.shape[1]) & (y < maskOther.shape[0])

                    masked_other = np.zeros_like(valid, dtype=bool)
                    masked_other[valid] = maskOther[y[valid], x[valid]] == 0
                    confidence_A_B[masked_other] = 0

                writer.write(pair_string, (confidence_A_B * 255.0).astype(np.uint8), np.uint8)
        except BaseException:
            writer.close()
            raise
        writer.close_and_link()

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
