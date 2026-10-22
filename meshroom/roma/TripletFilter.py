__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class TripletFilter(desc.CommandLineNode):
    """
    For each reference view, filter all tracks using triplet consistency.
    """

    category = "ROMA"
    size = avpar.DynamicViewsSize('inputSfMData')

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "tripletFilter.py").as_posix()
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
        desc.File(
            name="featuresFolder",
            label="Features folder",
            description="",
            value=""
        ),
        desc.File(
            name="matchesFolder",
            label="Matches folder",
            description="",
            value=""
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
