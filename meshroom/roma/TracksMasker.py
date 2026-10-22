__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class TracksMasker(desc.CommandLineNode):
    """
    Create a mask given tracks.
    For each track observation, add a circle of radius "radius".
    The goal is to avoid adding feature close to existing positions.
    """

    category = "ROMA"
    size = avpar.DynamicViewsSize('inputSfMData')

    parallelization = desc.Parallelization(blockSize=40)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    
    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "tracksMasker.py").as_posix()
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
            name="existingTracks",
            label="Tracks File",
            description="Use existing tracks to mask out pixels.",
            value=""
        ),
        desc.IntParam(
            name="radius",
            label="radius",
            description="Circles radius.",
            value=16,
            range=(0, 128, 1)
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output folder",
            description="",
            value="{nodeCacheFolder}"
        )
    ]
