__version__ = "1.2"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar

class RomaReducer(desc.CommandLineNode):

    category = "ROMA"
    documentation = """ RomaSampler is parallelized and distributed. It generated multiple
    output files which must merged together.
    Parallelized over views: each chunk writes the features of its views and the
    matches of the pairs where those views are the other image. """
    size = avpar.DynamicViewsSize('inputSfMData')

    parallelization = desc.Parallelization(blockSize=100)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "reducer.py").as_posix()
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
            name="samplesFolder",
            label="Samples folder",
            description="Samples folder from RomaSampler.",
            value="",
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
            name="featuresFolder",
            label="Output Features folder",
            description="",
            value="{nodeCacheFolder}"
        ),

        desc.File(
            name="matchesFolder",
            label="Output Matches folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
