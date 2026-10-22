from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat

import scipy.ndimage
import logging

from common import *

import os
import math
import h5py, hdf5plugin

def regionToNumpy(region):
    """
    Convert the feature positions of a region to a NumPy array.

    Args:
        region: An AliceVision region object exposing a ``Features()`` method
            that returns a sequence of objects with ``.x()`` and ``.y()`` accessors.

    Returns:
        numpy.ndarray: Float64 array of shape (N, 2) where each row is ``[x, y]``.
    """
    size = len(region.Features())
    array = np.empty(shape=(size, 2))

    vec = region.Features()
    for idx in range(0, size):

        f = vec[idx]
        array[idx, 0] = f.x()
        array[idx, 1] = f.y()

    return array

def compute_featuresMatcher(inputSfMData, imagePairsList, warpArchive, confidenceArchive, featuresFolder, matchesFolder, masksFolder, masksExtension, minConfidence, rangeIteration, rangeBlocksCount):
    """
    Match DSPSIFT features guided by a dense warp field.

    For each image pair in the assigned processing range the function:
    - loads DSPSIFT features for both images,
    - loads the dense warp and confidence EXR files,
    - optionally applies a binary reference mask to zero out confidence,
    - for every reference keypoint with sufficient confidence, restricts the
      search for matching other-image features to the warp-predicted region
      (using a spatial grid indexed on the other image),
    - selects the nearest-descriptor neighbour and applies a ratio test
      (second-best distance must be > 0.8 × best distance),
    - accumulates accepted matches into a global ``PairwiseMatches`` container
      and saves them as TXT files prefixed with the range iteration index.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        warpArchive (str): Archive containing warp arrays.
        confidenceArchive (str): Archive containing confidence arrays.
        featuresFolder (str): Directory containing ``.dspsift.feat`` / ``.dspsift.desc``
            feature files.
        matchesFolder (str): Directory where output match TXT files are written.
        masksFolder (str): Directory containing optional binary mask images.
            Pass an empty string to disable masking.
        masksExtension (str): File extension of the mask images (e.g. ``"png"``).
        minConfidence (float): Minimum warp confidence required to consider a
            reference keypoint for matching.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
     # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

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
    
    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]
    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")
        
    global_matches = avmatch.PairwiseMatches()

    for item in pairsToProcess:
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   
        
        #load features
        regionsRef = avfeat.SiftRegions()
        regionsOther = avfeat.SiftRegions()
        regionsRef.Load(f"{featuresFolder}/{referenceId}.dspsift.feat", f"{featuresFolder}/{referenceId}.dspsift.desc")
        regionsOther.Load(f"{featuresFolder}/{otherId}.dspsift.feat", f"{featuresFolder}/{otherId}.dspsift.desc")

        #load warp
        pair_string = str(referenceId) + "_" + str(otherId)
        with h5py.File(warpArchive, "r") as f_warp_h5, \
             h5py.File(confidenceArchive, "r") as f_conf_h5:
            if pair_string not in f_warp_h5 or pair_string not in f_conf_h5:
                logging.warning(f"Warp or confidence not found for pair {pair_string}, skipping.")
                continue
            warp_A_B = f_warp_h5[pair_string][()].astype(np.float32)
            confidence_A_B = f_conf_h5[pair_string][()].astype(np.float32) / 255.0

        W = confidence_A_B.shape[1]
        H = confidence_A_B.shape[0]

        #Load mask
        mask = None
        if len(masksFolder) > 0 :
            # Replace the extension with the mask extension
            stem = os.path.splitext(os.path.basename(referenceInfo.path))[0]
            mask_filename = f"{stem}.{masksExtension}"

            # Build the path to the correct mask
            path_mask = os.path.join(masksFolder, mask_filename)

            if os.path.exists(path_mask):
                maskLarge = open_image(path_mask, isBW=True, isFloat=False)
                maskSmall = avimage.Image_uchar()
                avimage.resampleImage(W, H, maskLarge, maskSmall, False);
                mask = maskSmall.getNumpyArray()

        #Apply mask if exists
        if mask is not None:
            if mask.shape[0] == warp_A_B.shape[0] and mask.shape[1] == warp_A_B.shape[1]:
                confidence_A_B[mask == 0] = 0

        #retrieve a list of coordinates for reference features
        refCoords = regionToNumpy(regionsRef)
        refCoords[:, 0] *= float(W) / float(referenceInfo.width)
        refCoords[:, 1] *= float(H) / float(referenceInfo.height)
        refCoordsInt = refCoords.astype(int)

        #retrieve a list of coordinates for other features
        otherCoords = regionToNumpy(regionsOther)
        otherCoords[:, 0] *= float(W) / float(otherInfo.width)
        otherCoords[:, 1] *= float(H) / float(otherInfo.height)
        otherCoordsInt = otherCoords.astype(int)

        grid = [[[] for i in range(H)] for j in range(W)]
        for row in range(otherCoords.shape[0]):
            ix = otherCoordsInt[row, 0]
            iy = otherCoordsInt[row, 1]
            grid[iy][ix].append(row)

        filtered_confidence = scipy.ndimage.minimum_filter(confidence_A_B, size=3, mode='constant', cval=np.inf)
        minimum_warp = scipy.ndimage.minimum_filter(warp_A_B, size=3, mode='constant', cval=np.inf)
        maximum_warp = scipy.ndimage.maximum_filter(warp_A_B, size=3, mode='constant', cval=-np.inf)

        matches = avmatch.IndMatches() 

        for row in range(refCoords.shape[0]):
            ix = refCoordsInt[row, 0]
            iy = refCoordsInt[row, 1]

            if ix < 1 or iy < 1:
                continue
            
            if ix >= (W - 1)  or iy >= (H - 1):
                continue

            if filtered_confidence[iy, ix] < minConfidence:
                continue

            #get region of search
            minx = minimum_warp[iy, ix, 0] * W
            miny = minimum_warp[iy, ix, 1] * H
            maxx = maximum_warp[iy, ix, 0] * W
            maxy = maximum_warp[iy, ix, 1] * H
            iminx = int(np.floor(minx))
            iminy = int(np.floor(miny))
            imaxx = int(np.ceil(maxx))
            imaxy = int(np.ceil(maxy))
            
            #extract features in region of search
            subgrid = grid[iminy:imaxy][iminx:imaxx]
            
            firstId = -1
            firstDist = 0
            secondId = -1
            secondDist = 0

            subgrid = [x for l in subgrid for x in l]
            subgrid = [x for l in subgrid for x in l]

            for otherid in subgrid:
                dist = regionsRef.SquaredDescriptorDistance(row, regionsOther, otherid)
                if dist < firstDist or firstId < 0:
                    firstDist = dist
                    firstId = otherid
                elif dist < secondDist or secondId < 0:
                    secondDist = dist
                    secondId = otherid
            
            if firstId < 0 or secondId < 0:
                continue
            
            if secondDist * 0.8 < firstDist:
                continue

            matches.append(avmatch.IndMatch(row, firstId))

        perdesc = avmatch.MatchesPerDescType()
        perdesc[avmatch.EImageDescriberType_DSPSIFT] = matches

        pair = avmatch.Pair(referenceId, otherId)
        global_matches[pair] = perdesc
    
    #Save all features and matches
    avmatch.Save(global_matches, matchesFolder, "txt", False, f"{rangeIteration}_")

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpArchive', type=str, help='')
    parser.add_argument('--confidenceArchive', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--masksFolder', type=str, help='')
    parser.add_argument('--masksExtension', type=str, help='')
    parser.add_argument('--minConfidence', type=float, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_featuresMatcher)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        args.func(inputSfMData=args.inputSfMData,
                imagePairsList=args.imagePairsList,
                warpArchive=args.warpArchive,
                confidenceArchive=args.confidenceArchive,
                featuresFolder=args.featuresFolder,
                matchesFolder=args.output,
                masksFolder=args.masksFolder,
                masksExtension=args.masksExtension,
                minConfidence=args.minConfidence,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()