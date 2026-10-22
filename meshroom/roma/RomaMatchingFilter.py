__version__ = "1.1"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class RomaMatchingFilter(desc.CommandLineNode):
    """Assumes a set of features have already been matched photometrically.
    Confirm the match using Roma.
    """

    category = "ROMA"
    size = avpar.DynamicViewsSize('inputSfMData')
    
    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "matchingFilter.py").as_posix()
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
            description="Input features",
            value=""
        ),
        desc.File(
            name="matchesFolder",
            label="Matches folder",
            description="Input matches",
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
