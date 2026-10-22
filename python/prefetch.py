"""Background loading of RomaPrepare views, ahead of the GPU.

Reading a view from the prepare archive on network storage takes ~100 ms for the
DINO features and ~250 ms for the images, which would otherwise stall the GPU on
every fine-cache miss. A loader thread reads the views in the order the matcher
will need them, with its own file handle, and keeps them in CPU memory (float16,
as stored) until the end of the chunk.
"""
import threading

import h5py

from prepare import load_view


class Prefetcher:
    """
    Load views of a prepare archive in a background thread.

    Args:
        index_path (str): Path of the RomaPrepare archive.
        view_order (list[int]): Views to load, in the order they will be needed (duplicates ignored).
        with_images (bool): Also load the resized images.
    """

    def __init__(self, index_path, view_order, with_images):
        self._order = list(dict.fromkeys(view_order))
        self._known = set(self._order)
        self._with_images = with_images
        self._loaded = {}
        self._error = None
        self._stop = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, args=(index_path,), daemon=True)
        self._thread.start()

    def _run(self, index_path):
        try:
            with h5py.File(index_path, "r") as index:
                for view_id in self._order:
                    if self._stop:
                        return
                    view = load_view(index, view_id, "cpu", with_images=self._with_images, half=True)
                    with self._cond:
                        self._loaded[view_id] = view
                        self._cond.notify_all()
        except BaseException as e:
            with self._cond:
                self._error = e
                self._cond.notify_all()

    def get(self, view_id):
        """
        Return a view, waiting for the loader if it has not reached it yet.

        Args:
            view_id (int): View id, one of ``view_order``.

        Returns:
            dict: As :func:`prepare.load_view` with ``device="cpu"`` and ``half=True``.

        Raises:
            KeyError: ``view_id`` is not in ``view_order``.
            Exception: The loader failed before reaching ``view_id``.
        """
        if view_id not in self._known:
            raise KeyError(view_id)
        with self._cond:
            while view_id not in self._loaded and self._error is None:
                self._cond.wait()
            if view_id in self._loaded:
                return self._loaded[view_id]
            raise self._error

    def close(self):
        """Stop the loader and release the loaded views."""
        self._stop = True
        self._thread.join()
        self._loaded.clear()
