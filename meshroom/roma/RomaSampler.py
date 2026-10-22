__version__ = "1.3"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class RomaSampler(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Sample the dense ROMA matches to generate features/matches used by SFM.
    This is an intermediate node which has to be followed by RomaReducer.
    """
    size = avpar.DynamicViewsSize('inputSfMData')

    parallelization = desc.Parallelization(blockSize=40)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    
    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "sampler.py").as_posix()
    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
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
        desc.IntParam(
            name="maxMatches",
            label="Requested matches",
            description="All Uncertainties from all pairs starting from the same view are mixed to enforce connections.",
            value=10000,
            range=(0, 50000, 1000)
        ),
        desc.FloatParam(
            name="minConfidence",
            label="Minimal confidence",
            description="Minimal confidence threshold.",
            value=0.15,
            range=(0.0, 1.0, 0.01)
        ),
        desc.IntParam(
            name="radiusMP",
            label="MP radius",
            description="Max Pooling radius. Only the best points in a circle of given radius will be kept.",
            value=16,
            range=(0, 128, 1)
        ),
        desc.File(
            name="filtersFolder",
            label="Filters Folder",
            description="Json files containing the estimated geometric filters",
            value=""
        ),
        desc.ChoiceParam(
            name="describerTypes",
            label="Describer Types",
            description="Describer types used to describe an image.",
            values=DESCRIBER_TYPES,
            value=["roma"],
            exclusive=False,
            joinChar=",",
            commandLineGroup="ignored"
        ),
    ]

    outputs = [
        desc.File(
            name="samplesFolder",
            label="Output Samples folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
