"""Byte-bounded LRU cache for per-image RoMa fine features."""
from collections import OrderedDict
from typing import Any, Callable


class FineCache:
    """LRU cache bounded by the total size in bytes of its items.

    Args:
        capacity_bytes: Maximum total size of cached items.
        compute: Builds the item for a view id on a cache miss.
        nbytes: Returns the size in bytes of an item.
    """

    def __init__(self, capacity_bytes: int, compute: Callable[[int], Any],
                 nbytes: Callable[[Any], int]):
        self._capacity = capacity_bytes
        self._compute = compute
        self._nbytes = nbytes
        self._items = OrderedDict()  # view_id -> (item, size)
        self._total = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, view_id: int) -> Any:
        """Return the item for a view, computing it on a miss.

        Least recently used items are evicted until the total fits; the item
        just requested is always kept, even if it alone exceeds the capacity.

        Args:
            view_id: Identifier of the view.

        Returns:
            The cached or newly computed item.
        """
        if view_id in self._items:
            self._items.move_to_end(view_id)
            return self._items[view_id][0]
        self.misses += 1
        item = self._compute(view_id)
        size = self._nbytes(item)
        self._items[view_id] = (item, size)
        self._total += size
        while self._total > self._capacity and len(self._items) > 1:
            _, (_, old_size) = self._items.popitem(last=False)
            self._total -= old_size
        return item


def capacity_from_vram(budget_bytes: int = 14 * 2**30,
                       reserve_bytes: int = 5 * 2**30) -> int:
    """Compute how many bytes the cache may use on the current GPU.

    Args:
        budget_bytes: Upper bound for total GPU memory use.
        reserve_bytes: Memory kept free for the matcher's working set.

    Returns:
        Cache capacity in bytes, never negative.
    """
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    return max(0, min(total, budget_bytes) - torch.cuda.memory_allocated() - reserve_bytes)
