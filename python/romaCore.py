"""Separable stages of the RoMa v2 pipeline.

`RoMaV2.match` runs everything per pair. These functions expose the same
computation split by stage so per-image work (preprocess, DINO, fine features)
can be cached and the pair-level coarse stage batched. The staged path
reproduces `model.match` exactly.
"""
import os

import torch
import torch.nn.functional as F
from romav2 import RoMaV2
from romav2.device import device
from romav2.features import Descriptor
from romav2.romav2 import _interpolate_warp_and_confidence, _map_confidence

# RoMa v2 settings supported here, see `RoMaV2.apply_setting`; the benchmark settings are left out
SETTINGS = ("turbo", "fast", "base", "precise")


def load_model(setting="precise"):
    """Load RoMa v2 in the given setting.

    Args:
        setting: One of :data:`SETTINGS`. It sets the low/high resolutions
            and whether a high resolution stage runs.

    Returns:
        The RoMaV2 model.
    """
    weights = None
    dinov3_path = None
    if "ROMATCH_MODELS_PATH" in os.environ:
        modelPath = os.environ["ROMATCH_MODELS_PATH"]
        weights = torch.load(os.path.join(modelPath, "romav2.0.1.pt"), weights_only=True)
        dinov3_path = os.path.join(modelPath, "dinov3")
    cfg = RoMaV2.Cfg(descriptor=Descriptor.Cfg(module_path=dinov3_path), weights=weights, compile=False)
    model = RoMaV2(cfg=cfg)
    model.apply_setting(setting)
    return model


def compile_supported():
    """Whether `torch.compile` of the refiners is supported on the current GPU.

    The refiners run under bf16 autocast; compiling them needs native bf16 (Ampere, sm 8.0+).

    Returns:
        bool: True on a CUDA device of compute capability 8.0 or newer.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


def _ensure_host_compiler():
    """Point inductor at gcc/g++ when CC/CXX name a compiler that is not installed (e.g. clang)."""
    import shutil
    for var, fallback in (("CC", "gcc"), ("CXX", "g++")):
        if shutil.which(os.environ.get(var, fallback)) is None:
            os.environ[var] = fallback


def enable_compile(model):
    """Compile the refiners' conv blocks with `torch.compile` if the GPU supports it.

    Measured −27% refiner time on an RTX A4000; outputs differ from eager mode at bf16
    rounding level. The first pair of each job pays the compilation time.

    Args:
        model: RoMaV2 model, modified in place.

    Returns:
        bool: True if the refiners were compiled.
    """
    if not compile_supported():
        return False
    _ensure_host_compiler()
    for refiner in model.refiners.values():
        refiner.block1 = torch.compile(refiner.block1, dynamic=False)
        refiner.hidden_blocks = torch.compile(refiner.hidden_blocks, dynamic=False)
    return True


def _check(model):
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Float32 matmul precision must be set to highest")
    model.eval()


def _resize(img, size):
    return F.interpolate(img, size=size, mode="bicubic", align_corners=False, antialias=True)


def _has_hr(model):
    return model.H_hr is not None and model.W_hr is not None


@torch.inference_mode()
def preprocess(model, pil):
    """Convert a PIL image to the low and high resolution inputs of RoMa.

    Args:
        model: RoMaV2 model, whose setting gives the resolutions.
        pil: PIL image.

    Returns:
        (img_lr, img_hr) float32 tensors (1,3,H,W) on the device; img_hr is
        None for settings without a high resolution stage.
    """
    img = RoMaV2._load_image(None, pil)
    img_hr = _resize(img, (model.H_hr, model.W_hr)) if _has_hr(model) else None
    return _resize(img, (model.H_lr, model.W_lr)), img_hr


@torch.inference_mode()
def dino(model, imgs_lr):
    """Run the DINO backbone.

    Args:
        model: RoMaV2 model.
        imgs_lr: (B,3,H,W) low resolution images.

    Returns:
        List of feature tensors (B,h,w,C).
    """
    _check(model)
    return model.f(imgs_lr)


@torch.inference_mode()
def coarse(model, fA, fB):
    """Run the bidirectional coarse matcher on a batch of pairs.

    Args:
        model: RoMaV2 model.
        fA: DINO features of the A images.
        fB: DINO features of the B images.

    Returns:
        Dict with warp_AB, confidence_AB, warp_BA, confidence_BA.
    """
    _check(model)
    # the matcher overwrites the last element of the lists it receives
    out = model.matcher(list(fA), list(fB), img_A=None, img_B=None, bidirectional=True, return_attention=False)
    return {k: out[k] for k in ("warp_AB", "confidence_AB", "warp_BA", "confidence_BA")}


@torch.inference_mode()
def fine(model, img_lr, img_hr=None):
    """Compute refiner features of one image at each resolution.

    Args:
        model: RoMaV2 model.
        img_lr: low resolution image (1,3,H,W).
        img_hr: high resolution image (1,3,H,W), or None without a high resolution stage.

    Returns:
        Tuple with one dict keyed by patch size per stage: (features_lr,) or
        (features_lr, features_hr).
    """
    _check(model)
    if img_hr is None:
        return (model.refiner_features(img_lr),)
    return model.refiner_features(img_lr), model.refiner_features(img_hr)


@torch.inference_mode()
def refine(model, c, fine_A, fine_B, directions=("AB", "BA")):
    """
    One or two-stage refinement of one pair, exactly as `RoMaV2.forward`.

    The A→B and B→A refinements are independent chains, so refining one
    direction only gives bit-identical results for that direction at half the cost.

    Args:
        model: RoMaV2 model.
        c: batch-1 slice of the output of `coarse`.
        fine_A: output of `fine` for image A; one stage per resolution.
        fine_B: output of `fine` for image B.
        directions: Directions to refine, subset of ("AB", "BA").

    Returns:
        Dict with ``warp_<d>`` and ``confidence_<d>`` for each requested direction.
    """
    _check(model)
    state = {d: (c[f"warp_{d}"], c[f"confidence_{d}"]) for d in directions}
    for stage in range(len(fine_A)):
        fA, fB = fine_A[stage], fine_B[stage]
        H, W = (model.H_lr, model.W_lr) if stage == 0 else (model.H_hr, model.W_hr)
        scale_factor = torch.tensor((W / model.anchor_width, H / model.anchor_height), device=device)
        for patch_size_str, refiner in model.refiners.items():
            ps = int(patch_size_str)
            zero = ps == 4 and stage == 1
            for d, (f_from, f_to) in (("AB", (fA, fB)), ("BA", (fB, fA))):
                if d not in state:
                    continue
                warp, conf = _interpolate_warp_and_confidence(
                    warp=state[d][0], confidence=state[d][1], H=H, W=W, patch_size=ps, zero_out_precision=zero)
                o = refiner(f_A=f_from[ps], f_B=f_to[ps], prev_warp=warp, prev_confidence=conf, scale_factor=scale_factor)
                state[d] = (o["warp"], o["confidence"])
    out = {}
    for d, (warp, conf) in state.items():
        out[f"warp_{d}"], out[f"confidence_{d}"] = warp, conf
    return out


@torch.inference_mode()
def finalize(preds):
    """Map confidences to overlap/precision and move everything to the CPU.

    Args:
        preds: output of `refine`.

    Returns:
        Dict with the same keys as `RoMaV2.match` (threshold None), for the refined directions only.
    """
    out = {}
    for d in ("AB", "BA"):
        if f"confidence_{d}" not in preds:
            continue
        overlap, precision = _map_confidence(confidence=preds[f"confidence_{d}"], threshold=None)
        out[f"warp_{d}"] = preds[f"warp_{d}"].cpu()
        out[f"confidence_{d}"] = preds[f"confidence_{d}"].cpu()
        out[f"overlap_{d}"] = overlap.cpu()
        out[f"precision_{d}"] = precision.cpu() if precision is not None else None
    return out
