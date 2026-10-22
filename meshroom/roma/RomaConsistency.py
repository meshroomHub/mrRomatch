__version__ = "2.0"

import os 
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import DESCRIBER_TYPES
from pyalicevision import parallelization as avpar


class RomaConsistency(desc.CommandLineNode):

    category = "ROMA"
    documentation = """
    Update confidence map based on consistency.
    The consistency is checked per triplet.
    A triplet is composed of one item (A) in the reference SfmData and two consecutive items (B,C) in the framesSfmData.
    framesSfmData is assumed to be a sequence.
    We check that the estimated cumulated motion A->B->C is close to the motion of A->C
    Warps and confidences are read from the RomaMatcher archives. Every reference->frame
    confidence found is written to the output archive, filtered when possible.
    """    

    size = avpar.DynamicViewsSize('referenceSfMData')
    parallelization = desc.Parallelization(blockSize=5)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"
    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "consistency.py").as_posix()
    commandLine="python "+exePath+" {allParams}"

    inputs = [
        desc.File(
            name="referenceSfMData",
            label="Reference SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="framesSfMData",
            label="Frames SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.ListAttribute(
            name="warpArchives",
            label="Warp Archives",
            description="List of warp archives where warps can be looked up.",
            elementDesc=desc.File(
                name="warpArchive",
                label="Warp Archive",
                description="",
                value=""
            ),
            exposed=True
        ),
        desc.ListAttribute(
            name="confidenceArchives",
            label="Confidence Archives",
            description="List of confidence archives where confidences can be looked up.",
            elementDesc=desc.File(
                name="confidenceArchive",
                label="Confidence Archive",
                description="",
                value=""
            ),
            exposed=True
        ),
        desc.FloatParam(
            name="maxDistance",
            label="Maximal distance for consistency",
            description="Minimal confidence threshold in pixels (in the size of the roma warp image).",
            value=32.0,
            range=(0.0, 64.0, 1.0)
        )
    ]

    outputs = [
        desc.File(
            name="outputConfidenceArchive",
            label="Output Confidence Archive",
            description="",
            value="{nodeCacheFolder}/confidence.h5"
        )
    ]
