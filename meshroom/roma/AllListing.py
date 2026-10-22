__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
import os

class AllListing(desc.Node):

    category = "ROMA"
    documentation = """ 
    Generate an imagePair list to match later. For all keyframes, matches with all the frames.
    """

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData.",
            value="",
        ),
        desc.File(
            name="keySfMData",
            label="Keyframes SfMData",
            description="Input Keyframes SfMData.",
            value="",
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        )
    ]

    outputs = [
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Path to a file which contains the list of image pairs to match.",
            value="{nodeCacheFolder}/imagesPairLists.txt"
        )
    ]

    def processChunk(self, chunk):

        from pyalicevision import sfmData as avsfmData
        from pyalicevision import sfmDataIO as avsfmDataIO
        from pyalicevision import matchingImageCollection as avmic

        chunk.logManager.start(chunk.node.verboseLevel.value)

        framesData = avsfmData.SfMData()
        ret = avsfmDataIO.load(framesData, chunk.node.inputSfMData.value, avsfmDataIO.VIEWS)
        if not ret:
            raise RuntimeError("Error with inputSfMData loading")

        keyframesData = avsfmData.SfMData()
        ret = avsfmDataIO.load(keyframesData, chunk.node.keySfMData.value, avsfmDataIO.VIEWS)
        if not ret:
            raise RuntimeError("Error with keySfMData data loading")

        kviews = keyframesData.getViews()
        views = framesData.getViews()

        plist = avmic.PairSet()
        for (kkey, kview) in kviews.items():
            for (key, view) in views.items():
                if kkey != key:
                    plist.append((kkey, key))
    
        if not avmic.savePairsToFile(chunk.node.imagePairsList.value, plist):
            raise RuntimeError("Error in image pairs list loading")

        chunk.logManager.end()
