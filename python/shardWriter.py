"""
Lock-free sharded HDF5 writing.

Each parallel job writes its own shard file ``<stem>.<chunk>.h5`` without any
lock, then links its keys into the shared index ``<stem>.h5`` as HDF5 external
links. Readers open the index and use ``key in f`` / ``f[key][()]`` as before.
Links store the shard basename, so they resolve relative to the index file's
directory regardless of the current working directory.
"""
import os
import time

import h5py
import hdf5plugin


def shard_path(index_path, chunk):
    """
    Build the shard file path for a chunk.

    Args:
        index_path (str): Path of the index file, ``<dir>/<stem>.h5``.
        chunk (int): Chunk (job) number.

    Returns:
        str: ``<dir>/<stem>.<chunk>.h5``.
    """
    stem, ext = os.path.splitext(index_path)
    return f"{stem}.{chunk}{ext}"


class ShardWriter:
    """
    Write datasets into a private shard and link them into a shared index.

    Args:
        index_path (str): Path of the shared index HDF5 file.
        chunk (int): Chunk number; selects the shard file. An existing shard
            is truncated.
    """

    def __init__(self, index_path, chunk):
        self.index_path = index_path
        self.shard = shard_path(index_path, chunk)
        self.keys = set()
        # Links to a shard that did not exist before this run cannot be stale.
        self._shard_existed = os.path.exists(self.shard)
        self._file = h5py.File(self.shard, "w")

    def write(self, key, data, dtype):
        """
        Create an LZ4-compressed dataset in the shard.

        Args:
            key (str): Dataset name.
            data (numpy.ndarray): Array to store.
            dtype: Stored NumPy dtype.
        """
        self._file.create_dataset(key, data=data, dtype=dtype, **hdf5plugin.LZ4(), chunks=True)
        self.keys.add(key)

    def write_group(self, key, arrays, dtype):
        """
        Create a group of LZ4-compressed datasets in the shard.

        Args:
            key (str): Group name.
            arrays (dict[str, numpy.ndarray]): Datasets to store by name.
            dtype: Stored NumPy dtype.
        """
        group = self._file.create_group(key)
        for name, array in arrays.items():
            group.create_dataset(name, data=array, dtype=dtype, **hdf5plugin.LZ4(), chunks=True)
        self.keys.add(key)

    def close(self):
        """
        Close the shard without linking it into the index (failed run).
        """
        self._file.close()

    def close_and_link(self):
        """
        Close the shard, then link its keys into the index under a lock.

        Existing index entries for these keys are replaced. The index is
        created if it does not exist.
        """
        self._file.close()
        lock_path = f"{self.index_path}.lock"
        # Atomic exclusive-create lock: flock is unreliable on NFS/Lustre.
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.1)
        try:
            with h5py.File(self.index_path, "a") as index:
                shard_name = os.path.basename(self.shard)
                if self._shard_existed:
                    # Drop links to this shard for keys this run no longer writes.
                    for key in list(index.keys()):
                        link = index.get(key, getlink=True)
                        if isinstance(link, h5py.ExternalLink) and link.filename == shard_name and key not in self.keys:
                            del index[key]
                for key in self.keys:
                    # getlink=True tests the link itself, not its (possibly dangling) target.
                    if index.get(key, getlink=True) is not None:
                        del index[key]
                    index[key] = h5py.ExternalLink(os.path.basename(self.shard), "/" + key)
        finally:
            os.unlink(lock_path)
