# -*- coding: utf-8 -*-
"""Backend implementations for motiflets distance computations."""

import math
from contextlib import contextmanager

import numpy as np
from numba import get_num_threads, njit, objmode, prange, set_num_threads, types
from numba.typed import Dict, List

from motiflets.distances import (
    sliding_mean_std,
    znormed_euclidean_distance,
    znormed_euclidean_distance_single,
)


# ---------------------------------------------------------------------------
# Thread control
# ---------------------------------------------------------------------------


@contextmanager
def numba_thread_context(n_jobs):
    """Temporarily adjust the Numba thread pool for backend kernels."""
    n_jobs = max(1, int(n_jobs))
    previous = get_num_threads()
    set_num_threads(n_jobs)
    try:
        yield previous
    finally:
        set_num_threads(previous)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True)
def _sliding_dot_product(query, time_series):
    """Compute a sliding dot-product using the Fourier Transform."""
    m = len(query)
    n = len(time_series)
    if m > n:
        raise ValueError("query longer than time_series")

    # Reverse query for cross-correlation and cast to float64
    q_rev = query[::-1]
    t = time_series

    # Next power-of-two ≥ n + m (good for FFT speed)
    total = n + m
    exponent = math.ceil(math.log2(total))
    L = 1 << exponent
    q_pad = np.concatenate((q_rev, np.zeros(L - m, dtype=q_rev.dtype)))
    t_pad = np.concatenate((t, np.zeros(L - n, dtype=t.dtype)))

    with objmode(conv="float64[:]"):
        conv = np.fft.irfft(np.fft.rfft(q_pad) * np.fft.rfft(t_pad))

    # Trim to the valid sliding-dot range
    return conv[m - 1: n]


@njit(nogil=True, fastmath=True, cache=True)
def _argknn(dist, k, m, lowest_dist=np.inf, slack=0.5):
    """Return up to k nearest neighbours subject to trivial match exclusion."""
    halve_m = np.int32(m * slack)

    dists = np.copy(dist)
    new_k = np.int32(min(len(dist) - 1, 2 * k))
    dist_pos = np.argpartition(dist, new_k)[:new_k]
    dist_sort = dist[dist_pos]

    idx = []
    for _ in np.arange(len(dist_sort)):
        p = np.argmin(dist_sort)
        pos = dist_pos[p]
        dist_sort[p] = np.inf

        if (
            (not np.isnan(dists[pos]))
            and (not np.isinf(dists[pos]))
            and (dists[pos] <= lowest_dist)
        ):
            idx.append(pos)
            dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf

        if len(idx) == k:
            break

    for _ in np.arange(len(idx), k):
        pos = np.argmin(dists)
        if (
            (not np.isnan(dists[pos]))
            and (not np.isinf(dists[pos]))
            and (dists[pos] <= lowest_dist)
        ):
            idx.append(pos)
            dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf
        else:
            break

    return np.array(idx, dtype=np.int32)


# ---------------------------------------------------------------------------
# Full backend (dense distance matrix)
# ---------------------------------------------------------------------------


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def compute_distances_with_knns_full(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
):
    """Compute the O(n²) distance matrix and k-NNs for each subsequence."""
    assert time_series.ndim == 2, "Dimensionality is not correct"

    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    n_jobs = max(1, min(n // 8, n_jobs))

    halve_m = np.int32(0)
    if exclude_trivial_match:
        halve_m = np.int32(m * slack)

    D = np.zeros((n, n), dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)
    bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))

    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min(start + bin_size, n)

        for d in range(dims):
            ts = time_series[d]
            preprocessing = distance_preprocessing(ts, m)
            dot_first = _sliding_dot_product(ts[:m], ts)

            dot_prev = None
            for order in range(start, end):
                if order == start:
                    dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    dot_rolled = (
                        np.roll(dot_prev, 1)
                        + ts[order + m - 1] * ts[m - 1:n + m]
                        - ts[order - 1] * np.roll(ts[:n], 1)
                    )
                    dot_rolled[0] = dot_first[order]

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)
                D[order] += dist
                dot_prev = dot_rolled

        for order in range(start, end):
            knn = _argknn(D[order], k, m, slack=slack)
            knns[order, :len(knn)] = knn

    return D, knns


# ---------------------------------------------------------------------------
# Sparse backend (selected pairs only)
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True)
def compute_upper_bound(
        ts, D_knn, knns, k, m,
        distance_single, preprocessing,
):
    """Estimate admissible extents to bound sparse distance evaluations."""
    kth_extent = np.zeros(k, dtype=np.float64)
    kth_extent[0] = np.inf

    for kk in np.arange(1, len(kth_extent)):
        best_knn_pos = np.argmin(D_knn[:, kk])
        candidate = knns[best_knn_pos, :kk + 1]
        kth_extent[kk] = get_pairwise_extent_raw(
            ts, candidate, m,
            distance_single,
            preprocessing)

        kth_nn_min = np.min(D_knn[:, kk])
        if kth_extent[kk] > 4 * kth_nn_min or kth_extent[kk] < kth_nn_min:
            kth_extent[kk] = kth_nn_min

    return kth_extent


@njit(fastmath=True, cache=True, nogil=True)
def get_pairwise_extent(D_full, motifset_pos, upperbound=np.inf):
    """Compute motif extent using a precomputed dense distance matrix."""
    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)

    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]

        for jj in range(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]

            motifset_extent = max(motifset_extent, D_full[i][j])
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(fastmath=True, cache=True, nogil=True)
def compute_distances_with_knns_sparse(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
):
    """Compute sparse distance storage alongside k-NNs for scalability."""
    assert time_series.ndim == 2, "Dimensionality is not correct"

    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    n_jobs = max(1, min(n // 8, n_jobs))

    halve_m = np.int32(0)
    if exclude_trivial_match:
        halve_m = np.int32(m * slack)

    bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))

    D_knn, knns = compute_distances_with_knns(
        time_series,
        m,
        k,
        exclude_trivial_match,
        n_jobs,
        slack,
        distance,
        distance_preprocessing,
    )

    preprocessing = []
    dot_first = []
    for dim in np.arange(time_series.shape[0]):
        preprocessing.append(distance_preprocessing(time_series[dim], m))
        dot_first.append(_sliding_dot_product(time_series[dim, :m], time_series[dim]))

    D_bool = [Dict.empty(key_type=types.int32, value_type=types.uint16) for _ in
              np.arange(n)]

    kth_extent = compute_upper_bound(
        time_series, D_knn, knns, k, m,
        distance_single, preprocessing,
    )

    for order in np.arange(0, n):
        for ks, dist_val in zip(knns[order], D_knn[order]):
            D_bool[order][ks] = True

            bound = False
            k_index = -1
            for kk in range(len(kth_extent) - 1, 0, -1):
                if D_knn[order, kk] <= kth_extent[kk]:
                    bound = True
                    k_index = kk + 1
                    break
            if bound:
                for ks2 in knns[order, :k_index]:
                    D_bool[ks][ks2] = True

    D_sparse = List()
    for _ in np.arange(n):
        D_sparse.append(Dict.empty(key_type=types.int32, value_type=types.float64))

    for idx in prange(n_jobs):
        dot_rolled = np.zeros((dims, n), dtype=np.float64)
        dot_prev = np.zeros((dims, n), dtype=np.float64)

        start = idx * bin_size
        end = min(start + bin_size, n)

        for order in range(start, end):
            dist_vec = np.zeros(n, dtype=np.float64)
            for d in range(dims):
                ts = time_series[d, :]
                if order == start:
                    dot_rolled[d] = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    dot_rolled[d] = (
                        np.roll(dot_prev[d], 1)
                        + ts[order + m - 1] * ts[m - 1:n + m]
                        - ts[order - 1] * np.roll(ts[:n], 1)
                    )
                    dot_rolled[d][0] = dot_first[d][order]

                dist_component = distance(
                    dot_rolled[d], n, m, preprocessing[d], order, halve_m
                )
                for i in range(len(dist_component)):
                    dist_vec[i] += dist_component[i]
                dot_prev[d] = dot_rolled[d]

            for key in D_bool[order]:
                D_sparse[order][key] = dist_vec[key]

    return D_sparse, knns


# ---------------------------------------------------------------------------
# Scalable backend (k-NNs only)
# ---------------------------------------------------------------------------


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def compute_distances_with_knns(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
):
    """Compute k-NNs for each subsequence with reduced memory footprint."""
    assert time_series.ndim == 2, "Dimensionality is not correct"

    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    n_jobs = max(1, min(n // 8, n_jobs))

    halve_m = np.int32(0)
    if exclude_trivial_match:
        halve_m = np.int32(m * slack)

    D_knn = np.zeros((n, k), dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)

    bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))

    preprocessing = []
    dot_first = []
    for d in np.arange(dims):
        ts = time_series[d]
        preprocessing.append(distance_preprocessing(ts, m))
        dot_first.append(_sliding_dot_product(ts[:m], ts))

    for idx in prange(n_jobs):
        dot_rolled = np.zeros((dims, n), dtype=np.float64)
        dot_prev = np.zeros((dims, n), dtype=np.float64)

        start = np.int32(idx * bin_size)
        end = np.int32(min(start + bin_size, n))

        for order in range(start, end):
            dist_vec = np.zeros(n, dtype=np.float64)
            for d in range(dims):
                ts = time_series[d, :]
                if order == start:
                    dot_rolled[d] = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    dot_rolled[d] = (
                        np.roll(dot_prev[d], 1)
                        + ts[order + m - 1] * ts[m - 1:n + m]
                        - ts[order - 1] * np.roll(ts[:n], 1)
                    )
                    dot_rolled[d][0] = dot_first[d][order]

                dist_component = distance(
                    dot_rolled[d], n, m, preprocessing[d], order, halve_m
                )
                for i in range(len(dist_component)):
                    dist_vec[i] += dist_component[i]
                dot_prev[d] = dot_rolled[d]

            knn = _argknn(dist_vec, k, m, slack=slack)
            for i in range(len(knn)):
                D_knn[order, i] = dist_vec[knn[i]]
                knns[order, i] = knn[i]

    return D_knn, knns


# ---------------------------------------------------------------------------
# Additional helpers
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True, nogil=True)
def get_pairwise_extent_raw(
        series, motifset_pos, motif_length,
        distance_single, preprocessing, upperbound=np.inf):
    """Compute motif extent via pairwise comparisons directly on the series."""
    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)

    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        a = series[:, i:i + motif_length]

        for jj in range(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]
            b = series[:, j:j + motif_length]

            dist_val = np.float64(0.0)
            for dim in range(series.shape[0]):
                dist_val += distance_single(a[dim], b[dim], i, j, preprocessing[dim])

            motifset_extent = max(motifset_extent, dist_val)
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(fastmath=True, cache=True, parallel=True)
def compute_distances_full(
        ts,
        m,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5):
    """Compute the full distance matrix for a univariate time series."""
    D, _ = compute_distances_with_knns_full(
        ts, m, k=1,
        exclude_trivial_match=exclude_trivial_match,
        n_jobs=n_jobs,
        slack=slack,
    )
    return D
