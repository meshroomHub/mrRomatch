import csv
import logging

import h5py
import hdf5plugin  # noqa: F401  (registers the LZ4 filter)
import numpy as np


def _read(f, key):
    """
    Read a dataset from an open HDF5 file.

    Args:
        f (h5py.File): Open archive.
        key (str): Dataset name.

    Returns:
        numpy.ndarray or None: The data, or None if the key is absent or unreadable.
    """
    try:
        if key in f:
            return f[key][()]
    except (KeyError, OSError):
        logging.warning(f"Cannot read {key}")
    return None


def compare(warp_a, conf_a, warp_b, conf_b, keys, min_conf=0.15):
    """
    Compare two warp/confidence archive pairs key by key.

    Args:
        warp_a (str): Warp archive A.
        conf_a (str): Confidence archive A.
        warp_b (str): Warp archive B.
        conf_b (str): Confidence archive B.
        keys (list[str] or None): Keys to compare; None means the union of keys of all archives.
        min_conf (float): Confidence threshold (conf_uint8 / 255 >= min_conf).

    Returns:
        list[dict]: One dict per key with ``key``, ``in_a``, ``in_b``, ``warp_px_mean``,
            ``warp_px_max`` (Euclidean xy difference in warp pixels over pixels confident in both,
            NaN if none), ``conf_abs_mean`` (mean |conf difference| in [0, 1] over all
            pixels, NaN if shapes differ), ``px_a`` and ``px_b`` (confident pixel counts).
    """
    rows = []
    with h5py.File(warp_a, "r") as wa, h5py.File(conf_a, "r") as ca, \
         h5py.File(warp_b, "r") as wb, h5py.File(conf_b, "r") as cb:
        if keys is None:
            keys = sorted(set(wa) | set(ca) | set(wb) | set(cb))
        for key in keys:
            wA, cA, wB, cB = _read(wa, key), _read(ca, key), _read(wb, key), _read(cb, key)
            row = {
                "key": key,
                "in_a": wA is not None and cA is not None,
                "in_b": wB is not None and cB is not None,
                "warp_px_mean": float("nan"),
                "warp_px_max": float("nan"),
                "conf_abs_mean": float("nan"),
                "px_a": 0,
                "px_b": 0,
            }
            if row["in_a"]:
                row["px_a"] = int(np.count_nonzero(cA.astype(np.float32) / 255.0 >= min_conf))
            if row["in_b"]:
                row["px_b"] = int(np.count_nonzero(cB.astype(np.float32) / 255.0 >= min_conf))
            if row["in_a"] and row["in_b"]:
                if wA.shape != wB.shape or cA.shape != cB.shape:
                    logging.warning(f"Shape mismatch for {key}")
                else:
                    ca_f = cA.astype(np.float32) / 255.0
                    cb_f = cB.astype(np.float32) / 255.0
                    row["conf_abs_mean"] = float(np.abs(ca_f - cb_f).mean())
                    mask = (ca_f >= min_conf) & (cb_f >= min_conf)
                    mask = mask.reshape(mask.shape[:2])
                    if mask.any():
                        # x and y are normalized, scale them to the warp size
                        scale = np.array([wA.shape[1], wA.shape[0]], dtype=np.float32)
                        d = ((wA[..., :2].astype(np.float32) - wB[..., :2].astype(np.float32)) * scale)[mask]
                        dist = np.sqrt((d ** 2).sum(axis=-1))
                        row["warp_px_mean"] = float(dist.mean())
                        row["warp_px_max"] = float(dist.max())
            rows.append(row)
    return rows


def write_csv(rows, path):
    """
    Write comparison rows to a CSV file.

    Args:
        rows (list[dict]): Rows returned by :func:`compare`.
        path (str): Output CSV path.
    """
    fields = ["key", "in_a", "in_b", "warp_px_mean", "warp_px_max", "conf_abs_mean", "px_a", "px_b"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(prog='compareArchives')
    parser.add_argument('--warpA', type=str, required=True, help='')
    parser.add_argument('--confA', type=str, required=True, help='')
    parser.add_argument('--warpB', type=str, required=True, help='')
    parser.add_argument('--confB', type=str, required=True, help='')
    parser.add_argument('--imagePairsList', type=str, help='Restrict the comparison to these pairs', default=None)
    parser.add_argument('--minConfidence', type=float, help='', default=0.15)
    parser.add_argument('--output', type=str, required=True, help='Output csv')
    args = parser.parse_args()

    keys = None
    if args.imagePairsList:
        from pyalicevision import matchingImageCollection as avmic
        plist = avmic.PairSet()
        if not avmic.loadPairsFromFile(args.imagePairsList, plist, False):
            raise RuntimeError("Error in image pairs list loading")
        keys = [f"{a}_{b}" for a, b in plist]
        logging.info(f"Loaded {len(keys)} pairs from list")

    rows = compare(args.warpA, args.confA, args.warpB, args.confB, keys, args.minConfidence)
    write_csv(rows, args.output)
    logging.info(f"Wrote {len(rows)} rows to {args.output}")
