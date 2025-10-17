# -*- coding: utf-8 -*-

import os
import time
from abc import ABC, abstractmethod
from warnings import simplefilter

import numpy as np
import psutil
from numba import set_num_threads, njit, prange, get_num_threads

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

STD_THRESHOLD = 1e-8


class BaseKnnBackend(ABC):
    """Abstract interface for approximate nearest-neighbour backends."""

    def __init__(self, *, m, k, search_radius, slack, n_jobs, verbose):
        self.m = m
        self.k = k
        self.search_radius = search_radius
        self.slack = slack
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _log(self, message):
        if self.verbose:
            print(message)

    @abstractmethod
    def compute(self, X_windows, process, previous_num_threads):
        """Return distances, timings, neighbours, and memory usage."""


class FaissBackend(BaseKnnBackend):
    """FAISS-based ANN backend."""

    def __init__(self, *, m, k, search_radius, slack, n_jobs, verbose, **kwargs):
        super().__init__(
            m=m, k=k,
            search_radius=search_radius, slack=slack,
            n_jobs=n_jobs, verbose=verbose
        )
        self.faiss_index = kwargs.get("faiss_index")
        self.M = kwargs.get("faiss_M", 64)
        self.efConstruction = kwargs.get("faiss_efConstruction", 500)
        self.efSearch = max(search_radius * k, kwargs.get("faiss_efSearch", 800))
        self.nlist = kwargs.get("faiss_nlist")

        if self.nlist is not None:
            self.nlist = int(self.nlist)

        self.nprobe = kwargs.get("faiss_nprobe", 32)
        self.nBits = kwargs.get("faiss_nBits", kwargs.get("nBits", 4))

    def compute(self, X_windows, process, previous_num_threads):
        import faiss

        faiss.omp_set_num_threads(self.n_jobs)

        start_time = time.time()
        quantizer = None
        index = None
        d = X_windows.shape[-1]

        if not self.faiss_index:
            raise ValueError(
                'faiss_index not set. Use "HNSW", "IVF", "IVFPQ", "IVFPQ+HNSW", "LSH".')

        if self.faiss_index == "LSH":
            n_bits = self.nBits * d
            self._log("\tLSH")
            self._log(f"\tnBits:       {n_bits}")
            index = faiss.IndexLSH(d, n_bits)

        elif self.faiss_index == "HNSW":
            self._log("\tHNSW")
            self._log(f"\tefSearch:       {self.efSearch}")
            self._log(f"\tefConstruction: {self.efConstruction}")
            self._log(f"\tM:              {self.M}")
            index = faiss.IndexHNSWFlat(d, self.M)
            index.hnsw.efConstruction = self.efConstruction
            index.hnsw.efSearch = self.efSearch

        elif self.faiss_index == "IVF":
            if not self.nlist:
                self.nlist = int(np.sqrt(X_windows.shape[0]))
            self._log("\tIVF")
            self._log(f"\tnlist:  {self.nlist}")
            self._log(f"\tnprobe: {self.nprobe}")
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_L2)
            index.train(X_windows)
            index.nprobe = self.nprobe

        elif self.faiss_index == "IVFPQ":
            if not self.nlist:
                self.nlist = int(np.sqrt(X_windows.shape[0]))
            self._log("\tIVFPQ")
            self._log(f"\tnlist:  {self.nlist}")
            self._log(f"\tnprobe: {self.nprobe}")
            mm = 64
            nbits = 8
            factory_string = f"IVF{int(self.nlist)},PQ{mm}x{nbits}"
            index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)
            index.train(X_windows)
            index.nprobe = self.nprobe

        elif self.faiss_index == "IVFPQ+HNSW":
            if not self.nlist:
                self.nlist = int(np.sqrt(X_windows.shape[0]))
            self._log("\tIVFPQ+HNSW")
            self._log(f"\tnlist:  {self.nlist}")
            self._log(f"\tnprobe: {self.nprobe}")
            self._log(f"\tefSearch:       {self.efSearch}")
            self._log(f"\tefConstruction: {self.efConstruction}")
            self._log(f"\tM:      {self.M}")
            mm = 64
            nbits = 8
            quantizer = faiss.IndexHNSWFlat(d, self.M)
            quantizer.hnsw.efConstruction = self.efConstruction
            quantizer.hnsw.efSearch = self.efSearch
            index = faiss.IndexIVFPQ(quantizer, d, self.nlist, mm, nbits,
                                     faiss.METRIC_L2)
            index.train(X_windows)
            index.nprobe = self.nprobe

        else:
            raise ValueError(
                f'Unknown FAISS index {self.faiss_index}. '
                'Use "HNSW", "IVF", "IVFPQ", "IVFPQ+HNSW", "LSH".')

        index.add(X_windows)
        index_create_time = time.time() - start_time

        index_search_time = time.time()
        D, knns = index.search(
            X_windows,
            self.search_radius * self.k
        )
        index_search_time = time.time() - index_search_time

        faiss.omp_set_num_threads(previous_num_threads)
        memory_usage = process.memory_info().rss / (1024 * 1024)

        if quantizer is not None:
            del quantizer
        del index

        return D, index_create_time, index_search_time, knns, memory_usage


class AnnoyBackend(BaseKnnBackend):
    """Annoy-based ANN backend."""

    def __init__(self, *, m, k, search_radius, slack, n_jobs, verbose, **kwargs):
        super().__init__(
            m=m, k=k,
            search_radius=search_radius, slack=slack,
            n_jobs=n_jobs, verbose=verbose)

        self.annoy_n_trees = kwargs.get("annoy_n_trees", 10)
        self.annoy_search_k = kwargs.get("annoy_search_k", -1)

    def compute(self, X_windows, process, _previous_num_threads):
        import annoy

        d = X_windows.shape[-1]
        start_time = time.time()
        index = annoy.AnnoyIndex(d, metric="euclidean")

        self._log("\tannoy")
        self._log(f"\tn_trees:  {self.annoy_n_trees}")
        self._log(f"\tsearch_k:  {self.annoy_search_k}")

        for i, vector in enumerate(X_windows):
            index.add_item(i, vector)

        index.build(self.annoy_n_trees, n_jobs=-1)
        index_create_time = time.time() - start_time

        index_search_time = time.time()
        knns = np.full((len(X_windows), self.k), -1, dtype=np.int32)
        D = np.full((len(X_windows), self.k), np.inf, dtype=np.float32)
        for i, vector in enumerate(X_windows):
            neighbors, distances = index.get_nns_by_vector(
                vector, self.k, self.annoy_search_k, include_distances=True)
            neighbors = np.asarray(neighbors, dtype=np.int32)
            distances = np.asarray(distances, dtype=np.float32)
            knns[i, :len(neighbors)] = neighbors
            D[i, :len(distances)] = distances

        index_search_time = time.time() - index_search_time

        memory_usage = process.memory_info().rss / (1024 * 1024)
        del index

        return D, index_create_time, index_search_time, knns, memory_usage


class PyNNDescentBackend(BaseKnnBackend):
    """PyNNDescent-based ANN backend."""

    def __init__(self, *, m, k, search_radius, slack, n_jobs, verbose, **kwargs):
        super().__init__(
            m=m, k=k,
            search_radius=search_radius, slack=slack,
            n_jobs=n_jobs, verbose=verbose
        )
        self.pynndescent_n_neighbors = kwargs.get("pynndescent_n_neighbors", 10)
        self.pynndescent_leaf_size = kwargs.get("pynndescent_leaf_size", 24)
        self.pynndescent_pruning_degree_multiplier = kwargs.get(
            "pynndescent_pruning_degree_multiplier", 1.0)
        self.pynndescent_diversify_prob = kwargs.get(
            "pynndescent_diversify_prob", 1.0)
        self.pynndescent_n_search_trees = kwargs.get(
            "pynndescent_n_search_trees", 1)
        self.pynndescent_search_epsilon = kwargs.get(
            "pynndescent_search_epsilon", 0.1)

    def compute(self, X_windows, process, _previous_num_threads):
        import pynndescent

        start_time = time.time()
        index = pynndescent.NNDescent(
            X_windows,
            metric="euclidean",
            low_memory=False,
            n_neighbors=self.pynndescent_n_neighbors,
            leaf_size=self.pynndescent_leaf_size,
            pruning_degree_multiplier=self.pynndescent_pruning_degree_multiplier,
            diversify_prob=self.pynndescent_diversify_prob,
            n_search_trees=self.pynndescent_n_search_trees,
            n_jobs=self.n_jobs
        )

        self._log("\tpynndescent")
        self._log(f"\tn_neighbors:  {self.pynndescent_n_neighbors}")
        self._log(f"\tleaf_size: {self.pynndescent_leaf_size}")
        self._log(f"\tpruning_degree_multiplier: "
                  f"{self.pynndescent_pruning_degree_multiplier}")
        self._log(f"\tdiversify_prob: {self.pynndescent_diversify_prob}")
        self._log(f"\tn_search_trees: {self.pynndescent_n_search_trees}")
        self._log(f"\tsearch_epsilon: {self.pynndescent_search_epsilon}")

        index_create_time = time.time() - start_time

        index_search_time = time.time()
        knns, D = index.neighbor_graph
        index_search_time = time.time() - index_search_time

        memory_usage = process.memory_info().rss / (1024 * 1024)
        del index

        return D, index_create_time, index_search_time, knns, memory_usage


BACKEND_REGISTRY = {
    "faiss": FaissBackend,
    "annoy": AnnoyBackend,
    "pynndescent": PyNNDescentBackend,
}

index_strategies = tuple(BACKEND_REGISTRY.keys())


class VectorSearchNearestNeighbors:
    """VectorSearch-based nearest neighbor computations for motiflet discovery."""

    def __init__(
            self,
            m,
            k,
            index_strategy="faiss",
            search_radius=5,
            slack=0.5,
            n_jobs=4,
            verbose=True,
            **kwargs):

        self.m = m
        self.k = k
        self.index_strategy = index_strategy
        self.search_radius = search_radius
        self.slack = slack
        requested_jobs = n_jobs
        if requested_jobs < 1:
            requested_jobs = os.cpu_count() or 1
        self.n_jobs = requested_jobs
        self.verbose = verbose

        self.backend = self._build_backend(dict(kwargs))

    def _build_backend(self, params):
        backend_cls = BACKEND_REGISTRY.get(self.index_strategy)

        if backend_cls is None:
            available = ", ".join(index_strategies)
            raise ValueError(
                f"Unknown indexing strategy: {self.index_strategy}. "
                f"Available strategies: {available}"
            )

        return backend_cls(
            m=self.m,
            k=self.k,
            search_radius=self.search_radius,
            slack=self.slack,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

    def compute_knns(self, X):
        """Computes approximate distances and k-nearest neighbors."""
        assert X.shape[0] == 1, \
            "Vector backends can handle univariate data, only."

        self.previous_jobs = get_num_threads()
        set_num_threads(self.n_jobs)

        pid = os.getpid()
        self.process = psutil.Process(pid)

        if X.ndim > 1:
            X = X.flatten()

        X_windows = znorm_windows(X, self.m)

        np.random.seed(42)
        np.random.shuffle(X_windows)

        (D_lb, index_create_time, index_search_time,
         knns_lb, memory_usage) = self.backend.compute(
            X_windows, self.process, self.previous_jobs)

        post_process_time = time.time()

        if self.verbose:
            print(f"\tApplying exclusion Zone")
            #print("\t", knns_lb[0])
            #print("\t", knns_lb[-1])

        D_exact, knns_exact = apply_exclusion_zone(
            X,
            self.m,
            D_lb,
            knns_lb,
            self.k,
            slack=self.slack
        )

        #if self.verbose:
        #    print("\t", knns_exact[0])
        #    print("\t", knns_exact[-1])

        post_process_time = time.time() - post_process_time

        set_num_threads(self.previous_jobs)

        print(f"Total time: "
              f"\n\tCreate: {index_create_time:.3f}s "
              f"\n\tSearch: {index_search_time:.3f}s "
              f"\n\tPost Process: {post_process_time:.3f}s.")

        return (D_exact, knns_exact, index_create_time,
                index_search_time, post_process_time, memory_usage)


@njit(fastmath=True, cache=True, inline='always')
def znormed_euclidean_distance_single(a, b, a_i, b_j, means, stds):
    """ Implementation of z-normalized Euclidean distance """
    m = len(a)
    z = 2 * m * (1 - (np.dot(a, b) - m * means[a_i] * means[b_j]) / (
            m * stds[a_i] * stds[b_j]))
    return np.sqrt(z)


@njit(fastmath=True, cache=True)
def sliding_mean_std(series, m):
    end = max(1, series.shape[0] - m + 1)
    stds = np.zeros(end)
    means = np.zeros(end)
    window = series[0:m]
    series_sum = np.sum(window)
    square_sum = np.sum(np.multiply(window, window))

    r_window_length = 1.0 / m
    mean = series_sum * r_window_length
    buf = np.sqrt(max(square_sum * r_window_length - mean * mean, 0.0))
    stds[0] = buf if buf > STD_THRESHOLD else 1
    means[0] = mean

    for w in range(1, end):
        series_sum += series[w + m - 1] - series[w - 1]
        mean = series_sum * r_window_length
        square_sum += (
                series[w + m - 1] * series[w + m - 1]
                - series[w - 1] * series[w - 1]
        )
        buf = np.sqrt(max(square_sum * r_window_length - mean * mean, 0.0))
        stds[w] = buf if buf > STD_THRESHOLD else 1
        means[w] = mean

    return means, stds


@njit(fastmath=True, cache=True)
def make_windows(X, window_size, n_chunks, chunk_size):
    X_windows = np.full((n_chunks, window_size), np.inf, dtype=X.dtype)
    for i in range(n_chunks):
        start = i * chunk_size
        X_windows[i] = X[start:start + window_size]
    return X_windows


def znorm_windows(X, window_size):
    # Apply windowing to the data, and z-normalize it
    num_inst = X.shape[0] - window_size + 1
    X_windows = X[np.arange(window_size)[None, :] + np.arange(num_inst)[:, None]]

    mean, std = sliding_mean_std(X, window_size)
    X_lb = (X_windows - mean[:, np.newaxis]) / std[:, np.newaxis]

    return X_lb


# FIXME: adding fastmath=True breaks the code???
@njit(cache=True, parallel=True)
def apply_exclusion_zone(X, m, D_lb, knns_lb, k, slack=0.5):
    """Go through the list of knns, any apply the exclusion zone.

    Returns the knns to each offset with exclusion applied
    """
    # compute size of the exclusion zone
    means, stds = sliding_mean_std(X, m)
    halve_m = np.int32(slack * m)

    n = D_lb.shape[0]

    D_knn = np.full((n, k), np.inf, dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)

    # The knns_lb - list can be overlapping, Thus apply slack (exclusion zone)
    # to take top-k neighbors
    for order in prange(n):
        dists = np.full(n, np.inf, dtype=np.float64)
        dists[knns_lb[order]] = D_lb[order]

        dist_pos = knns_lb[order]
        dist_sort = D_lb[order]

        # top-k counter
        k_idx = 0
        # go through the list, applying exclusion zone
        for i in range(len(dist_sort)):
            arg_pos = np.argmin(dist_sort)
            pos = dist_pos[arg_pos]
            d = dist_sort[arg_pos]
            dist_sort[arg_pos] = np.inf

            # check if the position is not within some exclusion zone to a previously
            # chosen index
            if ((dists[pos] != np.inf) and
                    (not np.isnan(dists[pos])) and
                    (not np.isinf(dists[pos]))):
                D_knn[order, k_idx] = d
                knns[order, k_idx] = np.int32(pos)

                # exclude all trivial matches and itself
                dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf
                k_idx += 1

            # We found the top-k elements
            if k_idx >= k:
                break

    D = np.zeros((n, D_knn.shape[-1]), dtype=np.float64)
    for i in prange(len(knns)):
        query = X[i:i + m]
        knn = knns[i]
        for a in prange(1, len(knn)):
            j = knn[a]
            if j > -1:
                # Re-rank based on z-normalized Euclidean distance
                D[i, a] = znormed_euclidean_distance_single(
                    query, X[j:j + m], i, j, means, stds)
            else:
                D[i, a] = np.inf

    return D, knns
