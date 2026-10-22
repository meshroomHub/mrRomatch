from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat
from pyalicevision import system as avsys

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

def compute_featuresMatcher(inputSfMData, imagePairsList, warpArchive, confidenceArchive, featuresFolder, matchesFolder, outputMatchesFolder, masksFolder, masksExtension, minConfidence, rangeIteration, rangeBlocksCount):
    """
    Filter existing feature matches using dense warp consistency.

    For each pair in the assigned processing range the function:
    - loads the pre-computed dense warp and confidence map,
    - optionally applies a binary mask to zero out confidence in masked regions,
    - for every existing match, samples the warp at the reference keypoint location
      and checks whether the predicted destination is within 4 pixels of the
      actual matched keypoint in the other image (using bilinear corner sampling),
    - retains only matches that pass both the confidence and distance checks.

    Filtered matches are saved to ``outputMatchesFolder`` as TXT files prefixed
    with the range iteration index.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        warpArchive (str): Archive containing warps arrays.
        confidenceArchive (str): Archive containing confidence arrays.
        featuresFolder (str): Directory containing feature ``.feat`` / ``.desc`` files.
        matchesFolder (str): Directory with the input match files to filter.
        outputMatchesFolder (str): Directory where filtered match files are written.
        masksFolder (str): Directory containing optional binary mask images.
            Pass an empty string to disable masking.
        masksExtension (str): File extension of the mask images (e.g. ``"png"``).
        minConfidence (float): Minimum warp confidence required to validate a match.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    pairsToProcess = list(plist)

    #Compute parallelization parameters using the number of pairs to process
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(pairsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return

    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]

    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    input_matches = avmatch.PairwiseMatches()
    output_matches = avmatch.PairwiseMatches()
    types = avmatch.EImageDescriberTypeVector()
    matches = avmatch.Load(input_matches, iinfos.keys(), [matchesFolder], types)
    if not matches:
        logging.error("Unable to load matches")
        raise RuntimeError()
    
    for (pairViews, matchesPerDescs) in input_matches.items():

        referenceId = pairViews[0]
        otherId = pairViews[1]

        if not pairViews in pairsToProcess:
            continue

        #load warp
        pair_string = str(referenceId) + "_" + str(otherId)
        with h5py.File(warpArchive, "r") as f_warp_h5, \
             h5py.File(confidenceArchive, "r") as f_conf_h5:
            if pair_string not in f_conf_h5 or pair_string not in f_warp_h5:
                continue
            warp_A_B = f_warp_h5[pair_string][()].astype(np.float32)
            confidence_A_B = f_conf_h5[pair_string][()].astype(np.float32) / 255.0

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

        scaleY = warp_A_B.shape[0] / iinfos[referenceId].height
        scaleX = warp_A_B.shape[1] / iinfos[referenceId].width

        perdesc = avmatch.MatchesPerDescType()

        for (desc, matchesPerDesc) in matchesPerDescs.items():

            #load features
            regionsRef = avfeat.SiftRegions()
            regionsOther = avfeat.SiftRegions()
            regionsRef.Load(f"{featuresFolder}/{referenceId}.{avfeat.EImageDescriberType_enumToString(desc)}.feat", f"{featuresFolder}/{referenceId}.{avfeat.EImageDescriberType_enumToString(desc)}.desc")
            regionsOther.Load(f"{featuresFolder}/{otherId}.{avfeat.EImageDescriberType_enumToString(desc)}.feat", f"{featuresFolder}/{otherId}.{avfeat.EImageDescriberType_enumToString(desc)}.desc")
            featuresRef = regionsRef.Features()
            featuresOther = regionsOther.Features()

            matches = avmatch.IndMatches() 

            for item in matchesPerDesc:
                pRef = featuresRef[item._i]
                pOther = featuresOther[item._j]

                ox = pOther.x() * scaleX
                oy = pOther.y() * scaleY
                rx = int(np.floor(pRef.x() * scaleX))
                ry = int(np.floor(pRef.y() * scaleY))

                minDist = 1e16
                for i in (0, 1):
                    for j in (0, 1):
                        destx = warp_A_B[ry+i, rx+j, 0] * warp_A_B.shape[0]
                        desty = warp_A_B[ry+i, rx+j, 1] * warp_A_B.shape[1]
                        conf = confidence_A_B[ry+i, rx+j]
                        if conf < minConfidence:
                            continue

                        dx = destx - ox
                        dy = desty - oy
                        dist = dx*dx + dy*dy
                        if dist < minDist:
                            minDist = dist

                #print(minDist)
                if np.sqrt(minDist) < 4:
                    matches.append(item)
            
            logging.info(f"pair {referenceId}, {otherId} has {len(matches)} matches.")
            perdesc[desc] = matches

        output_matches[pairViews] = perdesc
        
    #Save all features and matches
    avmatch.Save(output_matches, outputMatchesFolder, "txt", False, f"{rangeIteration}_")
                
        

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
    parser.add_argument('--matchesFolder', type=str, help='')
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
                matchesFolder=args.matchesFolder,
                outputMatchesFolder=args.output,
                masksFolder=args.masksFolder,
                masksExtension=args.masksExtension,
                minConfidence=args.minConfidence,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()