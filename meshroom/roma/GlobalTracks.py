__version__ = "1.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class GlobalTracks(desc.CommandLineNode):

    category = "ROMA"
    size = avpar.DynamicViewsSize('inputSfMData')

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "globalTracks.py").as_posix()
    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="existingTracks",
            label="Tracks File",
            description="Existing tracks to merge",
            value=""
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output Tracks File",
            description="",
            value="{nodeCacheFolder}/tracks.json"
        )
    ]
