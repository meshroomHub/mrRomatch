from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat

from common import *

import logging
import os

def export_features(regionsMap, idView, coords):
    """
    Append feature points to the regions container for a given view.

    For each coordinate whose confidence (third column) is greater than 1e-6,
    a new ``RomaDescriptor`` and a corresponding ``PointFeature`` are added to
    the region associated with ``idView``. Coordinates with zero or negligible
    confidence are skipped so that the feature count stays consistent with the
    match indices built later.

    Args:
        regionsMap (dict): Mapping from view ID to ``RomaRegions`` objects.
            The region for ``idView`` is updated in-place.
        idView (int): View identifier whose region should be updated.
        coords (numpy.ndarray): Array of shape (N, 4) where columns are
            ``[x, y, confidence, scale]`` in the original image resolution.

    Returns:
        int: Index of the first newly added feature, i.e. the feature count
            before this call. Use this offset when building match indices.
    """
    regionsRef = regionsMap[idView]
    start = regionsRef.RegionCount()
   
    count = 0
    for coord in coords:
        if coord[2] > 1e-6:
            regionsRef.Descriptors().append(avfeat.RomaDescriptor())
            regionsRef.Features().append(avfeat.PointFeature(coord[0], coord[1], coord[3], 0.0))
            count = count + 1
           
        
    regionsMap[idView] = regionsRef

    return start

def saveFeatures(regionsMap, outputFolder):
    """
    Save all region feature and descriptor files to disk.

    For each view in ``regionsMap`` the function writes two files inside
    ``outputFolder``: ``<viewId>.roma.feat`` (feature positions/scales) and
    ``<viewId>.roma.desc`` (descriptors).

    Args:
        regionsMap (dict): Mapping from view ID to ``RomaRegions`` objects.
        outputFolder (str): Directory where the feature and descriptor files
            are written.
    """
    for (key, region) in regionsMap.items():
        
        ffeat = f"{outputFolder}/{key}.roma.feat"
        fdesc = f"{outputFolder}/{key}.roma.desc"
        
        region.Save(ffeat, fdesc)

def reduce_samples(inputSfMData, imagePairsList, samplesFolder, featuresFolder, matchesFolder, rangeIteration, rangeBlocksCount):
    """
    Convert pre-computed sample files into AliceVision feature and match files.
    RomaSampler was parallelized, RomaReducer do the post processing reduction.

    Parallelization is done over views: each chunk owns a range of views and
    is the only one writing their ``.roma.feat`` / ``.roma.desc`` files. To
    avoid any dependency between chunks, the features of a view are laid out
    as:
    - first, all the keypoints of the view used as a reference
      (``<viewId>.npy``), so that the features of a reference always start at
      index 0,
    - then, for every pair where the view is the other image (sorted by
      reference id), the valid matched points from
      ``<referenceId>_<viewId>.npy``.

    The matches of a pair are therefore written by the chunk owning its other
    view, which is the only one knowing the offset of that pair's features.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        samplesFolder (str): Directory containing the ``.npy`` sample files
            produced by the sampler stage.
        featuresFolder (str): Directory where ``.roma.feat`` / ``.roma.desc``
            files are written.
        matchesFolder (str): Directory where the match TXT files are written.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    from pyalicevision import system as avsys

    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    pairsToProcess = list()
    for pair in plist:
        if not pair[0] in iinfos or not pair[1] in iinfos:
            continue
        pairsToProcess.append(pair)

    # build a list of image pairs indexed by their other images
    plistByOther = dict()
    for item in sorted(pairsToProcess, key=lambda p: (p[0], p[1])):
        plistByOther.setdefault(item[1], []).append(item)

    # Computing parallelization parameters
    viewsToProcess = sorted(iinfos.keys())
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(viewsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return
    viewsToProcess = viewsToProcess[rangeStart:rangeEnd]

    #Start features objects
    regionsMap = dict()
    for viewId in viewsToProcess:
        regionsMap[viewId] = avfeat.RomaRegions()

    global_matches = avmatch.PairwiseMatches()

    for viewId in viewsToProcess:
        logging.info(f"Processing view #{viewId}")

        # Load file with coordinates for this view used as a reference
        # This is an array of size [n, 4], where n is the number of coordinates used in this reference view
        # first column is the x coordinate
        # second column is the y coordinate
        # third column is the 1
        # fourth column is the scale
        # Those features come first, so matches with this view as reference use the row index directly
        path_coords = os.path.join(samplesFolder, str(viewId) + ".npy")
        try:
            coords_A_B = np.load(path_coords)
        except:
            coords_A_B = np.array(())
        export_features(regionsMap, viewId, coords_A_B)

        # For all pairs with the current view as the other view
        for item in plistByOther.get(viewId, []):

            referenceId = item[0]

            # Load file with precomputed matches for a given reference view
            # This is an array of size [n, 4], where n is the number of matches
            # first column is the x coordinates in the other image
            # second column is the y coordinates in the other image
            # third column is the confidence score
            # fourth column is the scale
            path_samples = os.path.join(samplesFolder, str(referenceId) + "_" + str(viewId) + ".npy")

            try:
                match_A_B = np.load(path_samples)
            except:
                continue

            # Transform this array to a set of feature points for this view
            # We return the offset in the list of features to be able to compute indices for matches
            otherOffset = export_features(regionsMap, viewId, match_A_B)

            # Build a list of matches using offset and indices
            # The reference features start at index 0 in the reference view
            pos = 0
            matches = avmatch.IndMatches()
            for rowId in range(0, match_A_B.shape[0]):
                if match_A_B[rowId, 2] > 1e-6:
                    matches.append(avmatch.IndMatch(rowId, otherOffset + pos))
                    pos = pos + 1

            # Save matches for pair per desc.
            # Obviously, here we only have on descriptor type
            perdesc = avmatch.MatchesPerDescType()
            perdesc[avmatch.EImageDescriberType_ROMA] = matches

            # Save matches for pair per desc in a global container
            pair = avmatch.Pair(referenceId, viewId)
            global_matches[pair] = perdesc

    #Save all features and matches of this chunk
    avmatch.Save(global_matches, matchesFolder, "txt", False, f"{rangeIteration}_")
    saveFeatures(regionsMap, featuresFolder)


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--samplesFolder', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--matchesFolder', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    
    parser.set_defaults(func=reduce_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                imagePairsList=args.imagePairsList,
                samplesFolder=args.samplesFolder,
                featuresFolder=args.featuresFolder,
                matchesFolder=args.matchesFolder,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
