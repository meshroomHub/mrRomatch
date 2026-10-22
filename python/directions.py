import numpy as np


def prepare_warp(w):
    """
    Convert a RoMa warp tensor to a NumPy array with normalized [0, 1] coordinates.

    The input tensor uses [-1, 1] coordinates (RoMa convention). This function
    rescales them to [0, 1] and appends a zero channel to produce a 3-channel
    array suitable for saving as an EXR image.

    Args:
        w (torch.Tensor): Warp tensor of shape (H, W, 2) with values in [-1, 1].

    Returns:
        numpy.ndarray: Float32 array of shape (H, W, 3) with xy coordinates in
            [0, 1] and a zero third channel.
    """
    w = ((w + 1.0) / 2.0).detach().cpu().numpy().copy()
    w = np.concatenate([w, np.zeros([w.shape[0], w.shape[1], 1], dtype=np.float32)], axis=-1)

    return w

def prepare_confidence(c):
    """
    Convert a RoMa confidence tensor to a NumPy array.

    Args:
        c (torch.Tensor): Confidence tensor of arbitrary shape.

    Returns:
        numpy.ndarray: Detached, CPU-side copy of the tensor as a NumPy array.
    """
    c = c.detach().cpu().numpy().copy()
    
    return c

def check_loop(warp_A_B, warp_B_A, dist_B_A):
    """
    Compute a zoom-tolerant loop-consistency error for each A pixel.

    For every pixel ``p`` in image A the function:

    1. Follows ``warp_A_B`` to find the landing position ``q`` in image B.
    2. Evaluates ``warp_B_A`` at each of the four nearest integer neighbours
       of ``q`` to back-project to image A.
    3. Measures the Euclidean distance from each back-projected position to
       ``p`` (the A→B→A round-trip error at that neighbour).
    4. If ``dist_B_A`` is provided, replaces each neighbour's round-trip
       distance with ``min(A→B→A error, dist_B_A[neighbour])``.  This
       introduces the B→A→B error at the landing spot as an alternative
       consistency criterion (see :func:`updateUncertaintyWithLoops` for the
       rationale).
    5. Returns the minimum of these (possibly combined) values across all
       four neighbours.

    Pixels whose forward coordinate falls outside the image are assigned an
    error of ``inf``.

    Args:
        warp_A_B (numpy.ndarray): Forward warp of shape (H, W, 2) with
            normalised [0, 1] coordinates mapping pixels from A to B.
            Returns ``None`` immediately if this argument is ``None``.
        warp_B_A (numpy.ndarray): Backward warp of shape (H, W, 2) with
            normalised [0, 1] coordinates mapping pixels from B to A.
            Returns ``None`` immediately if this argument is ``None``.
        dist_B_A (numpy.ndarray or None): Pre-computed loop-error map of
            shape (H, W, 1) giving the B→A→B round-trip error for each B
            pixel.  When provided, it serves as an escape hatch: a pixel
            whose A→B→A error is large can still be considered consistent
            if the B→A→B error at its landing spot is small.  Pass ``None``
            to evaluate the A→B→A error alone.

    Returns:
        numpy.ndarray or None: Float array of shape (H, W, 1) containing the
            per-pixel loop-consistency error in pixels, or ``None`` if either
            warp is ``None`` or the warp shapes are incompatible.
    """
    if warp_A_B is None or warp_B_A is None:
        return None

    if warp_A_B.shape != warp_B_A.shape:
        return None

    if dist_B_A is not None and warp_A_B.shape != dist_B_A.shape:
        return None

    height, width = warp_A_B.shape[0], warp_A_B.shape[1]

    # Pixel coordinates of where each A pixel maps to in B
    x_A_B = warp_A_B[:, :, 0] * width
    y_A_B = warp_A_B[:, :, 1] * height

    # Integer neighbors for sampling warp_B_A
    ixm = np.floor(x_A_B).astype(int)
    iym = np.floor(y_A_B).astype(int)
    ixp = ixm + 1
    iyp = iym + 1

    # Valid mask (before clipping): forward coordinate must lie inside the image
    # Neighbors past the last row/column are clipped back onto it
    valid = (ixm >= 0) & (iym >= 0) & (ixm < width) & (iym < height)

    # Clip to valid range to avoid out-of-bounds access
    ixm = np.clip(ixm, 0, width - 1)
    iym = np.clip(iym, 0, height - 1)
    ixp = np.clip(ixp, 0, width - 1)
    iyp = np.clip(iyp, 0, height - 1)

    # Original pixel coordinates (column = x, row = y)
    y_orig, x_orig = np.mgrid[0:height, 0:width].astype(np.float32)

    # Sample warp_B_A at each of the 4 neighbors and measure round-trip error
    def dist_at(iy, ix):
        back_x = warp_B_A[iy, ix, 0] * width
        back_y = warp_B_A[iy, ix, 1] * height
        ret = np.sqrt((back_x - x_orig) ** 2 + (back_y - y_orig) ** 2)
        if dist_B_A is not None:
            ret = np.minimum(dist_B_A[iy, ix], ret)
        return ret

    dist = np.minimum(
        np.minimum(dist_at(iym, ixm), dist_at(iym, ixp)),
        np.minimum(dist_at(iyp, ixm), dist_at(iyp, ixp)),
    )
    dist = np.where(valid, dist, np.inf)

    return dist[..., np.newaxis]

def updateUncertaintyWithLoops(warp_A_B, warp_B_A, confidence_A_B, confidence_B_A, threshold):
    """
    Zero out confidences for matches that fail a zoom-tolerant loop-closure check.

    A naive A→B→A round-trip check breaks under zoom: when image B is zoomed
    out relative to A (many A pixels land on the same B pixel), the B pixel
    maps back to only one A pixel, so most of those A pixels incur a large
    A→B→A error even though the matches are geometrically correct.

    To handle this, the check uses the B→A→B error at the landing spot as an
    escape hatch.  In the zoom-out scenario, the B pixel's own B→A→B error is
    small (it maps to a unique A pixel which maps back correctly), so the match
    is correctly accepted.  Conversely, when zooming in (B is higher-res than
    A), the A→B→A error is small, so the check is valid in the traditional
    sense.  A match is only rejected when both loop directions are
    inconsistent.

    The check proceeds in two steps:

    1. Compute the B→A→B round-trip error for every B pixel (``dist_B_A``).
    2. For every A pixel, compute
       ``min(A→B→A error, dist_B_A at the B landing spot)`` (``dist_A_B``).

    A pixel is marked invalid when ``dist_A_B > threshold``, i.e. when
    **both** its A→B→A error **and** the B→A→B error at its landing spot
    exceed the threshold.

    Both ``confidence_A_B`` and ``confidence_B_A`` are zeroed for invalid
    A-space pixels in-place.

    Args:
        warp_A_B (numpy.ndarray): Forward warp of shape (H, W, 2) with
            normalised [0, 1] coordinates mapping pixels from A to B.
        warp_B_A (numpy.ndarray): Backward warp of shape (H, W, 2) with
            normalised [0, 1] coordinates mapping pixels from B to A.
        confidence_A_B (numpy.ndarray): Confidence map of shape (H, W, C)
            for the A→B direction; modified in-place.
        confidence_B_A (numpy.ndarray): Confidence map of shape (H, W, C)
            for the B→A direction; modified in-place.
        threshold (float): Maximum acceptable round-trip error in pixels.
            A pixel is invalidated only when both loop errors exceed this value.
    """
    dist_B_A = check_loop(warp_B_A, warp_A_B, None)
    if dist_B_A is None:
        return

    dist_A_B = check_loop(warp_A_B, warp_B_A, dist_B_A)
    if dist_A_B is None:
        return

    invalid = (dist_A_B > threshold)[..., 0]  # shape (H, W)
    confidence_A_B[invalid] = 0.0
    confidence_B_A[invalid] = 0.0




def finalize_direction(warp_d, warp_o, overlap_d, check_loops, loop_threshold, min_confidence):
    """
    Post-process one match direction the way the original RoMa matcher did for the A->B key.

    Inputs are never mutated: ``prepare_warp``/``prepare_confidence`` copy, and
    the loop check runs on those copies. The opposite confidence only receives
    in-place zeroing from the loop check and is discarded, so a copy of the
    direction's own confidence stands in for it.

    Args:
        warp_d (torch.Tensor): Warp of direction d, (H, W, 2) in [-1, 1].
        warp_o (torch.Tensor): Warp of the opposite direction, (H, W, 2).
        overlap_d (torch.Tensor): Overlap of direction d, (H, W, 1).
        check_loops (bool): Apply the loop-closure filter.
        loop_threshold (float): Loop error threshold in pixels.
        min_confidence (float): Pixels below this confidence are zeroed.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Warp float16 (H, W, 3) in [0, 1]
            and confidence uint8 (H, W, 1).
    """
    w_d = prepare_warp(warp_d)
    c_d = prepare_confidence(overlap_d)
    if check_loops:
        w_o = prepare_warp(warp_o)
        updateUncertaintyWithLoops(w_d, w_o, c_d, c_d.copy(), loop_threshold)

    low = c_d[..., 0] < min_confidence
    w_d[low] = 0
    c_d[low] = 0
    return w_d.astype(np.float16), (c_d * 255.0).astype(np.uint8)
