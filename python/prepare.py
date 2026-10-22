"""
Per-image preparation for the fast RoMa matcher.

Each view is decoded, resized to the RoMa resolutions of the chosen setting and
run through the DINO backbone once. Results are stored as float16 in an HDF5
archive, one group per view named ``str(viewId)`` whose ``setting`` attribute
names the RoMa setting, so pair matching can skip this work.
"""
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
import torch

import romaCore
from common import get_imageinfos_from_sfmdata, open_image_to_pil
from shardWriter import ShardWriter


def _decode_batch(infos, view_ids):
    return [open_image_to_pil(infos[v].path) for v in view_ids]


def prepare_views(model, setting, infos, index_path, chunk, batch=1):
    """
    Compute and store the DINO features and resized images of views.

    Image decoding runs on a CPU thread one batch ahead of the GPU.

    Args:
        model: RoMaV2 model from :func:`romaCore.load_model`, in the setting stored in the archive.
        setting (str): RoMa setting of ``model``, stored on every view.
        infos (dict[int, ImageInfo]): Views to process, by view id.
        index_path (str): Path of the shared HDF5 index to link the views into.
        chunk (int): Chunk number, selecting the shard file.
        batch (int): Number of images run through DINO at once. Defaults to 1 because batched DINO is not bit-identical to single-image DINO.
    """
    view_ids = list(infos)
    batches = [view_ids[i:i + batch] for i in range(0, len(view_ids), batch)]
    writer = ShardWriter(index_path, chunk)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(_decode_batch, infos, batches[0]) if batches else None
        for n, ids in enumerate(batches):
            pils = pending.result()
            pending = pool.submit(_decode_batch, infos, batches[n + 1]) if n + 1 < len(batches) else None
            resized = [romaCore.preprocess(model, pil) for pil in pils]
            features = romaCore.dino(model, torch.cat([lr for lr, _ in resized]))
            for i, view_id in enumerate(ids):
                logging.info(f"Prepared {view_id}")
                arrays = {"dino0": features[0][i], "dino1": features[1][i], "img_lr": resized[i][0][0]}
                if resized[i][1] is not None:
                    arrays["img_hr"] = resized[i][1][0]
                writer.write_group(str(view_id), {k: v.cpu().numpy() for k, v in arrays.items()}, np.float16,
                                   attrs={"setting": setting})
    writer.close_and_link()


def load_view(index, view_id, device, with_images=False, half=False):
    """
    Load the prepared data of a view.

    Args:
        index (h5py.File): Open archive written by :func:`prepare_views`.
        view_id (int): View id.
        device: Torch device to place the tensors on.
        with_images (bool): Also load the resized images.
        half (bool): Keep the tensors in float16 as stored (half the memory); cast with ``.float()`` before use.

    Returns:
        dict: ``"dino"``, a list of two float32 (float16 if ``half``) tensors (1,h,w,1024), and,
            if ``with_images``, ``"img_lr"`` (1,3,H_lr,W_lr) and ``"img_hr"``
            (1,3,H_hr,W_hr) tensors; ``"img_hr"`` is None for settings without
            a high resolution stage.
    """
    group = index[str(view_id)]

    def get(name):
        t = torch.from_numpy(group[name][()]).to(device)
        return (t if half else t.float()).unsqueeze(0)

    out = {"dino": [get("dino0"), get("dino1")]}
    if with_images:
        out["img_lr"] = get("img_lr")
        out["img_hr"] = get("img_hr") if "img_hr" in group else None
    return out


def read_setting(index, view_ids):
    """
    Read the RoMa setting the views were prepared with.

    Archives written before settings were stored hold ``precise`` data.

    Args:
        index (h5py.File): Open archive written by :func:`prepare_views`.
        view_ids (iterable[int]): Views to check.

    Returns:
        str: The setting shared by the views.

    Raises:
        RuntimeError: The views were prepared with different settings.
    """
    settings = {index[str(v)].attrs.get("setting", "precise") for v in view_ids}
    if len(settings) > 1:
        raise RuntimeError(f"Views prepared with different RoMa settings: {sorted(settings)}")
    return settings.pop() if settings else "precise"


def compute_prepare(inputSfMData, setting, outputPrepareArchive, rangeIteration, rangeBlocksCount):
    """
    Prepare the views of one parallelization block.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        setting (str): RoMa setting, one of :data:`romaCore.SETTINGS`.
        outputPrepareArchive (str): Path of the HDF5 index to write.
        rangeIteration (int): Index of the current processing block.
        rangeBlocksCount (int): Total number of processing blocks.
    """
    from pyalicevision import system as avsys

    infos = get_imageinfos_from_sfmdata(inputSfMData)
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(infos))
    if not valid:
        logging.error("Range is out of bounds.")
        return
    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")
    view_ids = list(infos)[rangeStart:rangeEnd]
    logging.info("Loading model")
    model = romaCore.load_model(setting)
    prepare_views(model, setting, {v: infos[v] for v in view_ids}, outputPrepareArchive, rangeIteration)


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(prog='romaPrepare')
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--setting', type=str, help='', default="precise", choices=romaCore.SETTINGS)
    parser.add_argument('--outputPrepareArchive', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    args = parser.parse_args()

    compute_prepare(inputSfMData=args.inputSfMData,
                    setting=args.setting,
                    outputPrepareArchive=args.outputPrepareArchive,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
