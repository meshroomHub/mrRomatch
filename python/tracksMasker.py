from common import *

import logging
import os

def compute_masks(inputSfMData, outputDirectory, imagePairsList, existingTracks, radius, rangeIteration, rangeBlocksCount):
    """
    Generate per-image binary masks highlighting regions covered by tracked features.

    For each reference image in the processed range, the function locates all track
    observations, dilates them by a disk of the given radius, inverts the result
    (masked-out = tracked areas), and writes the mask as an EXR file named after
    the source image stem.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        outputDirectory (str): Directory where output EXR mask files are written.
        imagePairsList (str): Path to the file listing image pairs to process.
        existingTracks (str): Path to the existing tracks file. If empty, no tracks
            are loaded and no masks are produced.
        radius (int): Dilation radius (in pixels at 1280x1280 resolution) applied
            around each tracked feature point.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """

    from pyalicevision import system as avsys
    from pyalicevision import track as avtrack
    
    # Todo : I need to find a way to guess it automatically.
    romaWidth = 1280
    romaHeight = 1280

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
        if ref in plistByRef:
            plistByRef[ref].append(item)
        else:
            plistByRef[ref] = [item]
    refsToProcess = list(plistByRef)

    # Build list of tuples per views containing the tracks observations
    tracks = avtrack.TracksMap()
    tracksPerView = {}
    if len(existingTracks)>0:
        if avtrack.loadTracks(tracks, existingTracks):
            for _, track in tracks.items():
                for viewId, feat in track.featPerView.items():

                    # Normalize coordinates between 0 and 1
                    w = iinfos[viewId].width
                    h = iinfos[viewId].height
                    x = feat.coords[0] / w
                    y = feat.coords[1] / h

                    if viewId not in tracksPerView:
                        tracksPerView[viewId] = [(x, y)] 
                    else:
                        tracksPerView[viewId].append((x, y)) 
    
    # Parallelization is done by splitting pairs based on their reference image
    # We want to have access to all the pairs from the same reference

    # Computeing parallelization parameters
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(refsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return
        
    refsToProcess = refsToProcess[rangeStart:rangeEnd]
    for referenceId in refsToProcess:
        
        if not referenceId in tracksPerView:
            continue
        
        reference_iinfo = iinfos[referenceId]
        stem = os.path.splitext(os.path.basename(reference_iinfo.path))[0]
        mask_filename = f"{stem}.exr"

        pts = np.array(tracksPerView[referenceId])

        ix = np.round(pts[:, 0] * romaWidth).astype(int)
        iy = np.round(pts[:, 1] * romaHeight).astype(int)
        valid = (ix >= 0) & (ix < romaWidth) & (iy >= 0) & (iy < romaHeight)
        ix = ix[valid]
        iy = iy[valid]

        # impulses (vectorized set)
        imp = np.zeros((romaHeight, romaWidth, 1), dtype=bool)
        imp[iy, ix] = True

        # Compute offsets around 0 which are inside the disk
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        disk = (yy*yy + xx*xx) <= radius*radius
        offsets = np.argwhere(disk) - radius

        # binary dilation by OR-ing shifted impulses
        out = np.zeros_like(imp)
        for dy, dx in offsets:
            ys0 = max(0,  dy); ys1 = min(romaHeight, romaHeight + dy)
            xs0 = max(0,  dx); xs1 = min(romaWidth, romaWidth + dx)
            out[ys0:ys1, xs0:xs1] |= imp[ys0-dy:ys1-dy, xs0-dx:xs1-dx]

        #write output
        path_output = os.path.join(outputDirectory, mask_filename)
        save_image(path_output, (~out).astype(np.uint8) * 255, True)
        

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='tracksMasker')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--existingTracks', type=str, help='')
    parser.add_argument('--radius', type=int, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_masks)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                imagePairsList=args.imagePairsList,
                existingTracks=args.existingTracks,
                radius=args.radius,
                outputDirectory=args.output,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
