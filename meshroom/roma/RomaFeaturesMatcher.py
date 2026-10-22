__version__ = "1.2"

import os 
from pathlib import Path

from meshroom.core import desc
from pyalicevision import parallelization as avpar

class RomaFeaturesMatcher(desc.CommandLineNode):

    category = "ROMA"

    documentation = """
    Use the Roma 'warp' images to restrict the region used to match features.
    A region will only be matched to features in the other image given the warp image coordinates.
    """

    size = avpar.DynamicViewsSize('inputSfMData')
    cpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=40)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "featuresMatcher.py").as_posix()

    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="featuresFolder",
            label="Features folder",
            description="Input features",
            value=""
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Path to a file which contains the list of image pairs to match.",
            value="",
        ),
        desc.File(
            name="warpArchive",
            label="Warp Archive",
            description="",
            value=""
        ),
        desc.File(
            name="confidenceArchive",
            label="Confidence Archive",
            description="",
            value=""
        ),
        desc.File(
            name="masksFolder",
            label="Masks folder",
            description="",
            value=""
        ),
        desc.ChoiceParam(
            name="masksExtension",
            label="Mask File Extension",
            description="Mask file extension",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
        ),
        desc.FloatParam(
            name="minConfidence",
            label="Minimal confidence",
            description="Minimal confidence threshold.",
            value=0.15,
            range=(0.0, 1.0, 0.01)
        )
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
