from common import *

import logging
from scipy.spatial import cKDTree


def filter_matches(inputSfMData, existingTracks, outputFilename):
    
    from pyalicevision import track as avtrack

    # Todo : I need to find a way to guess it automatically.
    romaWidth = 1280
    romaHeight = 1280

    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    tracks = avtrack.TracksMap()
    if len(existingTracks)==0:
        return
    
    logging.info("Loading all tracks")
    if not avtrack.loadTracks(tracks, existingTracks):
        logging.error("Impossible to load existing tracks")
        raise RuntimeError()

    logging.info(f"Converting {len(tracks)} tracks to Roma scale")
    # Convert once for all coordinates to roma dimensions
    romaCoords = {}
    for trackId, track in tracks.items():
        
        for viewId, feat in track.featPerView.items():
            
            # Normalize coordinates between 0 and 1
            w = iinfos[viewId].width
            h = iinfos[viewId].height
            x = feat.coords[0, 0] * romaWidth / w
            y = feat.coords[1, 0] * romaHeight / h

            if trackId not in romaCoords:
                romaCoords[trackId] = {}
            romaCoords[trackId][viewId] = (x, y)

    logging.info("Building kd-trees.")
    # Build per-view lists of (trackId, x, y) and KD-trees
    tracksPerView = {}
    for trackId, coords in romaCoords.items():
        for viewId, (x, y) in coords.items():
            if viewId not in tracksPerView:
                tracksPerView[viewId] = []
            tracksPerView[viewId].append((trackId, x, y))

    trees = {
        viewId: cKDTree(np.array([(x, y) for _, x, y in pts]))
        for viewId, pts in tracksPerView.items()
    }

    # Collect for each track, for each other track, 
    # the set of view id for which the point are close to each other
    logging.info("Collecting similar measures.")
    closeInView = {}
    for trackId, coords in romaCoords.items():
        closeInView[trackId] = {}
        for viewId, (x, y) in coords.items():
            indices = trees[viewId].query_ball_point([x, y], r=5.0)
            for i in indices:
                otherId = tracksPerView[viewId][i][0]
                if otherId == trackId:
                    continue
                
                if otherId not in closeInView[trackId]:
                    closeInView[trackId][otherId] = set()    
                closeInView[trackId][otherId].add(viewId)

    valid = {trackId : True for trackId in tracks}


    # Easy part, remove duplicates
    for trackId in closeInView:
        if not valid[trackId]:
            continue

        trackLen = len(romaCoords[trackId])
        
        for otherId in closeInView[trackId]:
            if not valid[otherId]:
                continue

            otherLen = len(romaCoords[otherId])

            if trackLen >= otherLen:
                if otherLen == len(closeInView[otherId][trackId]):
                    # other track is fully contained in current track
                    # It is considered a full duplicate
                    # We keep the current track (arbitraty)
                    valid[otherId] = False


    # Medium part, merge tracks with all their common views considered equal
    for trackId in closeInView:
        if not valid[trackId]:
            continue

        refTrack = tracks[trackId]
        refViews = set(refTrack.featPerView.keys())

        # Sort candidates by number of close views (descending) 
        # so stronger merges go first
        sorted_others = sorted(
            [oid for oid in closeInView[trackId] if valid[oid]],
            key=lambda oid: len(closeInView[trackId][oid]),
            reverse=True
        )
        
        for otherId in sorted_others:

            otherTrack = tracks[otherId]            
            otherViews = set(otherTrack.featPerView.keys())

            intersectionViews = refViews & otherViews
            closeIntersection = intersectionViews & closeInView[trackId][otherId]

            if len(intersectionViews) == len(closeIntersection):
                # Merge: absorb views from otherTrack that are not already in refTrack,
                # then invalidate otherTrack. Update refViews so that any subsequent
                # merge candidate is checked against the already-grown reference track.
                for viewId in otherViews - refViews:
                    refTrack.featPerView[viewId] = otherTrack.featPerView[viewId]
                
                refViews = set(refTrack.featPerView.keys())
                valid[otherId] = False
            

                
    outputTracks = avtrack.TracksMap()
    for trackId in closeInView:
        if valid[trackId]:
            outputTracks[trackId] = tracks[trackId]

    logging.error(f"Output {len(outputTracks)} to {outputFilename}")
    if not avtrack.saveTracks(outputTracks, outputFilename):
        logging.error("Impossible to save output tracks")
        raise RuntimeError()

    

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--existingTracks', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.set_defaults(func=filter_matches)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    existingTracks=args.existingTracks,
                    outputFilename=args.output)
    else:
        parser.print_help()
