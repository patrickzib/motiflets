# -*- coding: utf-8 -*-

import os
import time
from warnings import simplefilter

import numpy as np
import psutil
from numba import set_num_threads, njit, prange, get_num_threads
from motiflets.distances import znormed_euclidean_distance_single, sliding_mean_std

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

STD_THRESHOLD = 1e-8

index_strategies = [
    "faiss",
    "annoy",
    "pynndescent"
]


class VectorSearchNearestNeighbors:
    """ VectorSearch-based nearest neighbor computations for motiflet discovery. """

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
        self.n_jobs = n_jobs
        self.n_jobs = os.cpu_count() if self.n_jobs < 1 else self.n_jobs

        self.verbose = verbose

        #### faiss
        self.faiss_index = None
        if "faiss_index" in kwargs:
            self.faiss_index = kwargs["faiss_index"]

        self.M = kwargs["faiss_M"] if "faiss_M" in kwargs else 64
        self.efConstruction = kwargs[
            "faiss_efConstruction"] if "faiss_efConstruction" in kwargs else 500
        self.efSearch = kwargs["faiss_efSearch"] if "faiss_efSearch" in kwargs else 800
        self.efSearch = max(search_radius * self.k, self.efSearch)

        # number of clusters/cells
        self.nlist = kwargs["faiss_nlist"] if "faiss_nlist" in kwargs else None
        if self.nlist:
            self.nlist = int(self.nlist)

        # number of cells to search
        self.nprobe = kwargs["faiss_nprobe"] if "faiss_nprobe" in kwargs else 32

        # number of bits used for hashing (resolution)
        self.nBits = kwargs["faiss_nbits"] if "faiss_nbits" in kwargs else 4

        #### annoy

        self.annoy_n_trees \
            = kwargs["annoy_n_trees"] if "annoy_n_trees" in kwargs else 10
        self.annoy_search_k \
            = kwargs["annoy_search_k"] if "annoy_search_k" in kwargs else -1

        #### pynndescent

        self.pynndescent_n_neighbors \
            = kwargs[
            "pynndescent_n_neighbors"] if "pynndescent_n_neighbors" in kwargs else 10
        self.pynndescent_leaf_size \
            = kwargs[
            "pynndescent_leaf_size"] if "pynndescent_leaf_size" in kwargs else 24
        self.pynndescent_pruning_degree_multiplier \
            = kwargs[
            "pynndescent_pruning_degree_multiplier"] if "pynndescent_pruning_degree_multiplier" in kwargs else 1.0
        self.pynndescent_diversify_prob \
            = kwargs[
            "pynndescent_diversify_prob"] if "pynndescent_diversify_prob" in kwargs else 1.0
        self.pynndescent_n_search_trees \
            = kwargs[
            "pynndescent_n_search_trees"] if "pynndescent_n_search_trees" in kwargs else 1
        self.pynndescent_search_epsilon \
            = kwargs[
            "pynndescent_search_epsilon"] if "pynndescent_search_epsilon" in kwargs else 0.1

    def compute_knns(self, X):
        """Computes approximate distances and k-nearest neighbors."""
        assert X.shape[0] == 1, \
            "Vector backends can handle univariate data, only."

        # Set the number of threads for Numba
        self.previous_jobs = get_num_threads()
        set_num_threads(self.n_jobs)

        pid = os.getpid()
        self.process = psutil.Process(pid)

        if X.ndim > 1:
            X = X.flatten()

        X_windows = znorm_windows(X, self.m)

        # We must shuffle
        np.random.seed(42)
        np.random.shuffle(X_windows)

        if self.index_strategy == "faiss":
            D, index_create_time, index_search_time, knns, memory_usage \
                = self.process_faiss(X_windows)

        elif self.index_strategy == "annoy":
            D, index_create_time, index_search_time, knns, memory_usage \
                = self.process_annoy(X_windows)

        elif self.index_strategy == "pynndescent":
            D, index_create_time, index_search_time, knns, memory_usage \
                = self.process_pynndescent(X_windows)

        else:
            raise ValueError(
                f"Unknown indexing strategy: {index_strategy}. "
                f"Available strategies: {index_strategies}"
            )

        # Post-process the results to filter out distances and neighbors
        post_process_time = time.time()

        if self.verbose:
            print(f"\tApplying exclusion Zone")
            #print("\t", knns[0])
            #print("\t", knns[-1])

        D_exact, knns_exact = apply_exclusion_zone(
            X,
            self.m,  # :window_size
            D,
            knns,
            self.k,
            slack=self.slack
        )

        if self.verbose:
            print("\t", knns_exact[0])
            print("\t", knns_exact[-1])

        post_process_time = time.time() - post_process_time
        # print(f"\tPost-processing took {post_process_time:.3f} seconds.")

        # Set old values
        set_num_threads(self.previous_jobs)

        print(f"Total time: "
              f"\n\tCreate: {index_create_time:.3f}s "
              f"\n\tSearch: {index_search_time:.3f}s "
              f"\n\tPost Process: {post_process_time:.3f}s.")

        return (D_exact, knns_exact, index_create_time,
                index_search_time, post_process_time, memory_usage)

    def process_annoy(self, X_windows):
        import annoy

        d = X_windows.shape[-1]
        # https://github.com/spotify/annoy
        index_create_time = time.time()
        index = annoy.AnnoyIndex(d, metric="euclidean")

        if self.verbose:
            print(f"\tannoy")
            print(f"\tn_trees:  {self.annoy_n_trees}")
            print(f"\tsearch_k:  {self.annoy_search_k}")

        for i, X in enumerate(X_windows):
            index.add_item(i, X)

        index.build(self.annoy_n_trees, n_jobs=-1)
        index_create_time = time.time() - index_create_time

        index_search_time = time.time()
        # FIXME: no method to query multiple samples at the same time
        #        thus, too slow
        knns = np.zeros((len(X_windows), self.k), dtype=np.int32)
        D = np.zeros((len(X_windows), self.k), dtype=np.float32)
        for i, X in enumerate(X_windows):
            knns[i], D[i] = index.get_nns_by_vector(
                X_windows, self.k, self.annoy_search_k, include_distances=True)

        index_search_time = time.time() - index_search_time

        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # MB

        del index
        return D, index_create_time, index_search_time, knns, memory_usage

    def process_pynndescent(self, X_windows):
        import pynndescent

        # https://pynndescent.readthedocs.io/en/latest/api.html
        index_create_time = time.time()
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
            # compressed=True,
            # verbose=True,
        )

        if self.verbose:
            print(f"\tpynndescent")
            print(f"\tn_neighbors:  {self.pynndescent_n_neighbors}")
            print(f"\tleaf_size: {self.pynndescent_leaf_size}")
            print(
                f"\tpruning_degree_multiplier: {self.pynndescent_pruning_degree_multiplier}")
            print(f"\tdiversify_prob: {self.pynndescent_diversify_prob}")
            print(f"\tn_search_trees: {self.pynndescent_n_search_trees}")
            print(f"\tsearch_epsilon: {self.pynndescent_search_epsilon}")

        index_create_time = time.time() - index_create_time

        # We can then extract the nearest neighbors of each training sample by
        # using the neighbor_graph attribute.
        index_search_time = time.time()
        knns, D = index.neighbor_graph
        index_search_time = time.time() - index_search_time

        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # MB

        del index
        return D, index_create_time, index_search_time, knns, memory_usage

    def process_faiss(self, X_windows):

        import faiss
        faiss.omp_set_num_threads(self.n_jobs)

        # Compute distances using the lower bounding representation
        index_create_time = time.time()
        d = X_windows.shape[-1]

        if self.faiss_index:

            if self.faiss_index == "LSH":
                # setup our HNSW parameters
                n_bits = self.nBits * d  # total number of bits

                # number of neighbours we add to each vertex
                if self.verbose:
                    print(f"\tLSH")
                    print(f"\tnBits:       {n_bits}")

                index = faiss.IndexLSH(d, n_bits)

            elif self.faiss_index == "HNSW":
                # setup our HNSW parameters

                # number of neighbours we add to each vertex
                if self.verbose:
                    print(f"\tHNSW")
                    print(f"\tefSearch:       {self.efSearch}")
                    print(f"\tefConstruction: {self.efConstruction}")
                    print(f"\tM:              {self.M}")

                index = faiss.IndexHNSWFlat(d, self.M)
                index.hnsw.efConstruction = self.efConstruction
                index.hnsw.efSearch = self.efSearch  # TODO: reset every time needed?

            elif self.faiss_index == "IVF":
                # setup our IVF parameters

                if not self.nlist:
                    # number of clusters/cells set to sqrt(n)
                    self.nlist = int(np.sqrt(X_windows.shape[0]))

                if self.verbose:
                    print(f"\tIVF")
                    print(f"\tnlist:  {self.nlist}")
                    print(f"\tnprobe: {self.nprobe}")

                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_L2)
                index.train(X_windows)

                index.nprobe = self.nprobe  # TODO: reset every time needed?

            elif self.faiss_index == "IVFPQ":
                # setup our IVF-PQ parameters

                if not self.nlist:
                    # number of clusters/cells set to sqrt(n)
                    self.nlist = int(np.sqrt(X_windows.shape[0]))

                if self.verbose:
                    print(f"\tIVFPQ")
                    print(f"\tnlist:  {self.nlist}")
                    print(f"\tnprobe: {self.nprobe}")

                mm = 64  # d // 32  # Use 1/32 dimension for PQ
                nbits = 8  # bits per Subvector

                factory_string = f"IVF{int(self.nlist)},PQ{mm}x{nbits}"
                index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)
                index.train(X_windows)

                index.nprobe = self.nprobe  # TODO: reset every time needed?

            elif self.faiss_index == "IVFPQ+HNSW":
                # setup our IVF-PQ parameters

                if not self.nlist:
                    # number of clusters/cells set to sqrt(n)
                    self.nlist = int(np.sqrt(X_windows.shape[0]))

                if self.verbose:
                    print(f"\tIVFPQ+HNSW")
                    print(f"\tnlist:  {self.nlist}")
                    print(f"\tnprobe: {self.nprobe}")
                    print(f"\tefSearch:       {self.efSearch}")
                    print(f"\tefConstruction: {self.efConstruction}")
                    print(f"\tM:      {self.M}")

                mm = 64  # d // 32  # Use 1/32 dimension for PQ
                nbits = 8  # bits per Subvector

                # The coarse quantizer is responsible for finding the partition
                # centroids that are nearest to the query vector so that vector search
                # only needs to be performed on those partitions.
                quantizer = faiss.IndexHNSWFlat(D, self.M)
                quantizer.hnsw.efConstruction = self.efConstruction
                quantizer.hnsw.efSearch = self.efSearch

                index = faiss.IndexIVFPQ(quantizer, D, self.nlist, mm, nbits,
                                         faiss.METRIC_L2)

                index.train(X_windows)
                index.nprobe = self.nprobe  # TODO: reset every time needed?

            else:
                raise ValueError(
                    'Unknown FAISS index' + faiss_index + '.' +
                    'Use "HNSW", "IVF", "IVFPQ", "LSH".')
        else:
            raise ValueError(
                'faiss_index not set. Use "HNSW", "IVF", "IVFPQ", "LSH".')

        index.add(X_windows)
        index_create_time = time.time() - index_create_time

        # Find k-nearest neighbors based on the lower bound distances
        index_search_time = time.time()
        D, knns = index.search(
            X_windows,
            self.search_radius * self.k
        )

        index_search_time = time.time() - index_search_time
        # print(f"\tIndexing search took {index_search_time:.3f} seconds.")

        faiss.omp_set_num_threads(self.previous_jobs)
        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # MB

        # cleanup
        if 'quantizer' in locals():
            del quantizer
        del index

        return D, index_create_time, index_search_time, knns, memory_usage


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
                    query, X[j:j + m], i, j, (means, stds))
            else:
                D[i, a] = np.inf

    return D, knns
