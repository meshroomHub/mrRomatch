__version__ = "1.1"

from pathlib import Path
from meshroom.core import desc
from pyalicevision import parallelization as avpar

class RomaPrepare(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Compute once per image the resized inputs and DINO features used by the ROMA matcher.
    The RoMa setting chosen here is stored in the archive and used by RomaMatcher.
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
        desc.ChoiceParam(
            name="setting",
            label="Setting",
            description="RoMa v2 setting, from fastest to most precise:\n"
                        " - turbo: 320 px, single stage.\n"
                        " - fast: 512 px, single stage.\n"
                        " - base: 640 px, single stage.\n"
                        " - precise: 800 px refined at 1280 px.\n"
                        "The warps and confidences have the size of the last stage.",
            values=["turbo", "fast", "base", "precise"],
            value="precise",
            exclusive=True,
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
