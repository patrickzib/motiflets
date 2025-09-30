# -*- coding: utf-8 -*-

import os
import time
from warnings import simplefilter

import numpy as np
from numba import set_num_threads, njit, prange, get_num_threads

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

STD_THRESHOLD = 1e-8

index_strategies = [
    "faiss"
]


@njit(fastmath=True, cache=True, inline='always')
def znormed_euclidean_distance_single(a, b, a_i, b_j, means, stds):
    """ Implementation of z-normalized Euclidean distance """
    m = len(a)
    z = 2 * m * (1 - (np.dot(a, b) - m * means[a_i] * means[b_j]) / (
            m * stds[a_i] * stds[b_j]))
    return np.sqrt(z)


@njit(fastmath=True, cache=True, parallel=True)
def argsort_rows_topk(D, k):
    n_rows, _ = D.shape
    result = np.zeros((n_rows, k), dtype=np.int32)
    for i in prange(n_rows):
        for j in range(k):
            result[i, j] = np.argmin(D[i])
            D[i, result[i, j]] = np.inf  # set to inf to not select again
        # result[i] = np.argsort(D[i])[:k]
    return result


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
    D = np.zeros((n, knns_lb.shape[-1]), dtype=np.float64)

    for i in prange(len(knns_lb)):
        query = X[i:i + m]
        knn = knns_lb[i]

        for a in prange(1, len(knn)):
            j = knn[a]

            # Re-rank based on z-normalized Euclidean distance
            D[i, a] = znormed_euclidean_distance_single(
                query, X[j:j + m], i, j, means, stds)


    D_knn = np.full((n, k), np.inf, dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)

    # The knns_lb - list can be overlapping, Thus apply slack (exclusion zone)
    # to take top-k neighbors
    for order in prange(n):
        dists = np.full(n, np.inf, dtype=np.float64)
        dists[knns_lb[order]] = D[order]

        dist_pos = knns_lb[order]
        dist_sort = D[order]

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

    return D_knn, knns


def compute_knns_vector_search(
        X,
        m,
        k,
        index_strategy="faiss",
        search_radius=5,
        slack=0.5,
        n_jobs=4,
        **kwargs
):
    """Computes approximate distances and k-nearest neighbors using a lower bound."""
    assert X.shape[0] == 1, \
        "Vector backends can handle univariate data, only."

    # Set the number of threads for Numba
    n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
    previous_jobs = get_num_threads()
    set_num_threads(n_jobs)

    # if X.dtype != np.float32:
    #    X = X.astype(np.float32)

    if X.ndim > 1:
        X = X.flatten()

    X_windows = znorm_windows(X, m)

    # We must shuffle
    np.random.seed(42)
    np.random.shuffle(X_windows)

    if index_strategy == "faiss":
        import faiss
        faiss.omp_set_num_threads(n_jobs)

        # Compute distances using the lower bounding representation
        index_create_time = time.time()
        d = X_windows.shape[-1]

        if "faiss_index" in kwargs:
            faiss_index = kwargs["faiss_index"]
            if faiss_index == "HNSW":
                # setup our HNSW parameters

                # number of neighbours we add to each vertex
                M = kwargs["faiss_M"] if "faiss_M" in kwargs else 64
                # M = max(M, search_radius * k)

                efConstruction = kwargs["faiss_efConstruction"] if "faiss_efConstruction" in kwargs else 500
                efSearch = kwargs["faiss_efSearch"] if "faiss_efSearch" in kwargs else 800
                efSearch = max(search_radius * k, efSearch)

                print(f"\tefSearch:       {efSearch}")
                print(f"\tefConstruction: {efConstruction}")
                print(f"\tM:              {M}")

                index = faiss.IndexHNSWFlat(d, M)
                index.hnsw.efConstruction = efConstruction
                index.hnsw.efSearch = efSearch     # TODO: reset every time needed?

            elif faiss_index == "IVF":
                # setup our IVF parameters

                # number of clusters/cells
                nlist = kwargs["faiss_nlist"] if "faiss_nlist" in kwargs \
                    else np.sqrt(X_windows.shape[0])

                # number of cells to search
                nprobe = kwargs["faiss_nprobe"] if "faiss_nprobe" in kwargs \
                    else 32

                print(f"\tnlist:  {int(nlist)}")
                print(f"\tnprobe: {nprobe}")

                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss.METRIC_L2)
                index.nprobe = nprobe     # TODO: reset every time needed?
                index.train(X_windows)

            #elif faiss_index == "IVFPQ":
            #    # setup our IVFPQ parameters
            #    # TODO !!!
            #    pass
            else:
                raise ValueError(
                    'Unknown FAISS index' + faiss_index + '.' +
                    'Use "HNSW", "IVF", "IVFPQ".')
        else:
            raise ValueError(
                'faiss_index not set. Use "HNSW", "IVF", "IVFPQ".')


        index.add(X_windows)
        index_create_time = time.time() - index_create_time

        # Find k-nearest neighbors based on the lower bound distances
        index_search_time = time.time()
        D, knns = index.search(
            X_windows,
            search_radius * k    # FIXME: use M instead?
        )

        index_search_time = time.time() - index_search_time
        # print(f"\tIndexing search took {index_search_time:.3f} seconds.")

        faiss.omp_set_num_threads(previous_jobs)

        # cleanup
        if 'quantizer' in locals():
            del quantizer
        del index
    else:
        raise ValueError(
            f"Unknown indexing strategy: {index_strategy}. "
            f"Available strategies: {index_strategies}"
        )

    # Post-process the results to filter out distances and neighbors
    post_process_time = time.time()

    D_exact, knns_exact = apply_exclusion_zone(
        X,
        m,     # :window_size
        D,
        knns,
        k,
        slack=slack
    )
    print("\t", knns_exact[0])
    print("\t", knns_exact[-1])

    post_process_time = time.time() - post_process_time
    # print(f"\tPost-processing took {post_process_time:.3f} seconds.")

    # Set old values
    set_num_threads(previous_jobs)

    print(f"Total time: "
          f"\n\tCreate: {index_create_time:.3f}s "
          f"\n\tSearch: {index_search_time:.3f}s "
          f"\n\tPost Process: {post_process_time:.3f}s.")

    return D_exact, knns_exact, index_create_time, index_search_time, post_process_time
