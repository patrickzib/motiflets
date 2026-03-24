import numpy as np
from numba import int32, float64
from numba.experimental import jitclass

_maxheap_spec = [
    ("heap_dist", float64[:]),
    ("heap_candidates", int32[:, :]),
    ("size", int32),
    ("capacity", int32),
]


@jitclass(_maxheap_spec)
class MaxHeap:
    """Fixed-size max-heap for motiflet candidates.

    Public methods: push, replace_at, sorted_entries.
    Helper methods are prefixed with an underscore.
    """
    def __init__(self, capacity, k):
        self.heap_dist = np.full(capacity, np.inf, dtype=np.float64)
        self.heap_candidates = np.empty((capacity, k), dtype=np.int32)
        self.size = 0
        self.capacity = capacity

    def push(self, dist, candidate):
        i = self.size
        self.heap_dist[i] = dist
        self.heap_candidates[i] = candidate
        self.size += 1
        self._sift_up(i)

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap_dist[parent] >= self.heap_dist[i]:
                break
            self._swap(parent, i)
            i = parent

    def _sift_down(self, i):
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i

            if left < self.size and self.heap_dist[left] > self.heap_dist[largest]:
                largest = left
            if right < self.size and self.heap_dist[right] > self.heap_dist[largest]:
                largest = right

            if largest == i:
                break

            self._swap(i, largest)
            i = largest

    def _swap(self, a, b):
        self.heap_dist[a], self.heap_dist[b] = self.heap_dist[b], self.heap_dist[a]
        tmp = self.heap_candidates[a].copy()
        self.heap_candidates[a] = self.heap_candidates[b]
        self.heap_candidates[b] = tmp

    def sorted_entries(self):
        sorted_dists = np.full(self.capacity, np.inf, dtype=np.float64)
        sorted_candidates = np.full(self.heap_candidates.shape, -1, dtype=np.int32)
        if self.size == 0:
            return sorted_candidates, sorted_dists

        order = np.argsort(self.heap_dist[:self.size])
        for i in range(self.size):
            sorted_candidates[i] = self.heap_candidates[order[i]]
            sorted_dists[i] = self.heap_dist[order[i]]
        return sorted_candidates, sorted_dists

    def replace_at(self, position, dist, candidate):
        self.heap_dist[position] = dist
        self.heap_candidates[position] = candidate
        self._sift_down(position)
