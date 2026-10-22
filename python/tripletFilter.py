from common import *

from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import logging
import h5py, hdf5plugin
import numpy as np
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat


def _process_pair(args):
    """Worker: compute consistent (t1, t2) feature-key pairs for one image pair."""
    pair, features, warpArchive, confidenceArchive, tau = args
    pair_string = str(pair[0]) + "_" + str(pair[1])
    result = set()

    with h5py.File(warpArchive, "r") as f_warp_h5, \
         h5py.File(confidenceArchive, "r") as f_conf_h5:

        if pair_string not in f_warp_h5 or pair_string not in f_conf_h5:
            return result

        warp = f_warp_h5[pair_string][()].astype(np.float32)

        H, W = warp.shape[0], warp.shape[1]

        # Feature coordinates are normalized, scale them to the warp size
        coords = np.array([[f[0][2], f[0][3], f[1][2], f[1][3]] for f in features], dtype=np.float32)
        coords *= np.array([W, H, W, H], dtype=np.float32)

        ixm = np.floor(coords[:, 0]).astype(int)
        iym = np.floor(coords[:, 1]).astype(int)
        ixp = ixm + 1
        iyp = iym + 1

        # Neighbors past the last row/column are clipped back onto it
        in_bounds = (ixm >= 0) & (iym >= 0) & (ixm < W) & (iym < H)

        ixm_c = np.clip(ixm, 0, W - 1)
        iym_c = np.clip(iym, 0, H - 1)
        ixp_c = np.clip(ixp, 0, W - 1)
        iyp_c = np.clip(iyp, 0, H - 1)

        corner_iy = np.stack([iym_c, iym_c, iyp_c, iyp_c])
        corner_ix = np.stack([ixm_c, ixp_c, ixm_c, ixp_c])

        px = warp[corner_iy, corner_ix, 0] * W
        py = warp[corner_iy, corner_ix, 1] * H
        dx = px - coords[np.newaxis, :, 2]
        dy = py - coords[np.newaxis, :, 3]
        dists = np.sqrt(dx * dx + dy * dy)

        best_idx = np.argmin(dists, axis=0)
        n_idx = np.arange(len(features))
        best_dist = dists[best_idx, n_idx]
        consistent = in_bounds & (best_dist < tau)

        for i, (t1, t2) in enumerate(features):
            if consistent[i]:
                result.add(((t1[0], t1[1]), (t2[0], t2[1])))

    return result

def filter_matches(inputSfMData, imagePairsList, warpArchive, confidenceArchive, featuresFolder, matchesFolder, output):
    
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
    
    # build a list of image pairs indexed by their reference images
    plistByRef = dict()
    for item in pairsToProcess:
        ref = item[0]
        plistByRef.setdefault(ref, []).append(item)
    
    refsToProcess = list(plistByRef)

    # Load existing matches
    input_matches = avmatch.PairwiseMatches()
    types = avmatch.EImageDescriberTypeVector()
    matches = avmatch.Load(input_matches, iinfos.keys(), [matchesFolder], types)
    

    # Tuple is (viewId, featureId, x, y)
    # dictTracks is a dictionnary whose keys are Tuple and values are vectors of Tuple.
    # For each referencePoints, list the associated otherPoints matched in other nodes.
    dictTracks = dict()

    # Cache loaded features per (viewId, desc) to avoid redundant disk reads
    features_cache = dict()
    
    def get_features(viewId, desc):
        key = (viewId, desc)
        if key not in features_cache:
            desc_str = avfeat.EImageDescriberType_enumToString(desc)
            regions = avfeat.SiftRegions()
            regions.Load(f"{featuresFolder}/{viewId}.{desc_str}.feat", f"{featuresFolder}/{viewId}.{desc_str}.desc")
            
            # Normalized coordinates, scaled to each warp size when checked
            scaley = 1.0 / iinfos[viewId].height
            scalex = 1.0 / iinfos[viewId].width

            features_cache[key] = np.array([[f.x() * scalex, f.y() * scaley] for f in regions.Features()], dtype=np.float32)

        return features_cache[key]

    logging.info("Building reference centered point information.")
    for (pairViews, matchesPerDescs) in input_matches.items():

        referenceId = pairViews[0]
        otherId = pairViews[1]

        if not pairViews in pairsToProcess:
            continue

        for (desc, matchesPerDesc) in matchesPerDescs.items():

            featuresRef = get_features(referenceId, desc)
            featuresOther = get_features(otherId, desc)

            for match in matchesPerDesc:

                # Normalized points
                rx = featuresRef[match._i, 0]
                ry = featuresRef[match._i, 1]
                ox = featuresOther[match._j, 0]
                oy = featuresOther[match._j, 1]

                # Store the otherTuple to the list of Tuples attached to the reference point
                refTuple = (referenceId, match._i, rx, ry)
                otherTuple = (otherId, match._j, ox, oy)
                dictTracks.setdefault(refTuple, []).append(otherTuple)
    
    # Create a list of pairs of points to check per pair of images
    # Indexing per pair of images enable to factorize loading of warp images
    pairsPerReference = dict()
    logging.info("Building pairsPerReference.")
    for referenceTuple, vecOtherTuples in dictTracks.items():
        for t1, t2 in combinations(vecOtherTuples, 2):
            viewPair = (t1[0], t2[0])
            featureIds = (t1, t2)
            pairsPerReference.setdefault(viewPair, []).append(featureIds)

    # Triplet distance threshold (pixels of the warp image)
    tau = 4.0

    # Directed leaf-leaf pairs that passed the triangle constraint.
    # Each entry is ((viewId1, featIdx1), (viewId2, featIdx2)).
    consistent_pairs = set()

    # Read content from warp
    logging.info("Building consistent_pairs.")
    args_list = [
        (pair, features, warpArchive, confidenceArchive, tau)
        for pair, features in pairsPerReference.items()
    ]
    with ProcessPoolExecutor() as executor:
        for partial_pairs in executor.map(_process_pair, args_list):
            consistent_pairs.update(partial_pairs)

    # Per-star pruning: for each star (reference point + its matched leaves),
    # iteratively remove the leaf with the lowest triangle-support ratio until
    # all remaining leaves satisfy the threshold, then discard stars that are
    # too small to be trustworthy.
    #
    # minRatioSupport – fraction of other leaves in the star that must form a
    #                   consistent triangle (either warp direction) with a leaf.
    # minStarSize     – minimum number of surviving leaves to keep any match.
    minRatioSupport = 0.4
    minStarSize = 2

    # Set of (ref_key, leaf_key) edges that survived per-star pruning.
    #   ref_key  = (refViewId,   refFeatIdx)
    #   leaf_key = (otherViewId, otherFeatIdx)
    surviving_matches = set()
    logging.info("Building surviving_matches.")
    for referenceTuple, vecOtherTuples in dictTracks.items():
        ref_key = (referenceTuple[0], referenceTuple[1])
        leaves = list(vecOtherTuples)

        while True:
            n = len(leaves)
            if n < 2:
                # Cannot form any triangle; discard all leaves in this star.
                leaves = []
                break

            # Count how many other leaves form a consistent triangle with each
            # leaf, accepting either warp direction as evidence.
            support = []
            for i, t_i in enumerate(leaves):
                key_i = (t_i[0], t_i[1])
                s = sum(
                    1 for j, t_j in enumerate(leaves) if i != j and (
                        (key_i, (t_j[0], t_j[1])) in consistent_pairs or
                        ((t_j[0], t_j[1]), key_i) in consistent_pairs
                    )
                )
                support.append(s)

            possible = n - 1
            min_support = min(support)

            if min_support / possible >= minRatioSupport:
                break  # All leaves meet the threshold; star is clean.

            # Remove the single weakest leaf and iterate.
            leaves.pop(support.index(min_support))

        if len(leaves) >= minStarSize:
            for t in leaves:
                surviving_matches.add((ref_key, (t[0], t[1])))

    # Keep only pairwise matches where the specific ref->leaf edge survived pruning.
    output_matches = avmatch.PairwiseMatches()
    logging.info("Building output_matches.")
    for (pairViews, matchesPerDescs) in input_matches.items():
        if pairViews not in pairsToProcess:
            continue

        referenceId = pairViews[0]
        otherId = pairViews[1]
        perdesc = avmatch.MatchesPerDescType()

        for (desc, matchesPerDesc) in matchesPerDescs.items():
            filtered = avmatch.IndMatches()
            for match in matchesPerDesc:
                ref_key  = (referenceId, match._i)
                leaf_key = (otherId,     match._j)
                if (ref_key, leaf_key) in surviving_matches:
                    filtered.append(match)
            if len(filtered) > 0:
                perdesc[desc] = filtered

        if len(perdesc) > 0:
            output_matches[pairViews] = perdesc

    logging.info("Saving output_matches.")
    avmatch.Save(output_matches, output, "txt", False, "")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpArchive', type=str, help='')
    parser.add_argument('--confidenceArchive', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--matchesFolder', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.set_defaults(func=filter_matches)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpArchive=args.warpArchive,
                    confidenceArchive=args.confidenceArchive,
                    featuresFolder=args.featuresFolder,
                    matchesFolder=args.matchesFolder,
                    output=args.output)
    else:
        parser.print_help()
