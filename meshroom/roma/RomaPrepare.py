__version__ = "1.0"

from pathlib import Path
from meshroom.core import desc
from pyalicevision import parallelization as avpar

class RomaPrepare(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Compute once per image the resized inputs and DINO features used by the ROMA matcher.
    """
    size = avpar.DynamicViewsSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=50)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "prepare.py").as_posix()

    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
    ]

    outputs = [
        desc.File(
            name="outputPrepareArchive",
            label="Output Prepare Archive",
            description="Archive holding the prepared data of every view.",
            value="{nodeCacheFolder}/prepare.h5"
        ),
    ]
