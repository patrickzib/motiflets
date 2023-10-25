# -*- coding: utf-8 -*-
"""Compute k-Motiflets.


"""

__author__ = ["patrickzib"]

import itertools
from ast import literal_eval
from os.path import exists

import numpy as np
import numpy.fft as fft
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, prange, objmode, types
from numba.typed import Dict, List
from scipy.signal import argrelextrema
from scipy.stats import zscore
from tqdm.auto import tqdm


def as_series(data, index_range, index_name):
    """Coverts a time series to a series with an index.

    Parameters
    ----------
    data : array-like
        The time series raw data as numpy array
    index_range :
        The index to use
    index_name :
        The name of the index to use (e.g. time)

    Returns
    -------
    series : PD.Series

    """
    series = pd.Series(data=data, index=index_range)
    series.index.name = index_name
    return series


def _resample(data, sampling_factor=10000):
    """Resamples a time series to roughly `sampling_factor` points.

    The method searches a factor to skip every i-th point.

    Parameters
    ----------
    data : array-like
        The time series data
    sampling_factor :
        The rough size of the time series after sampling

    Returns
    -------
    Tuple
        data :
            The raw data after sampling
        factor : int
            The factor used to sample the time series

    """
    factor = 1
    if len(data) > sampling_factor:
        factor = np.int32(len(data) / sampling_factor)
        data = data[::factor]
    return data, factor


def read_ground_truth(dataset):
    """Reads the ground-truth data for the time series.

    Parameters
    ----------
    dataset : String
        Name of the dataset

    Returns
    -------
    Series : pd.Series
        A series of ground-truth data

    """
    file = '../datasets/ground_truth/' + dataset.split(".")[0] + "_gt.csv"
    if exists(file):
        print(file)
        series = pd.read_csv(file, index_col=0)

        for i in range(0, series.shape[0]):
            series.iloc[i] = series.iloc[i].apply(literal_eval)

        return series
    return None


def read_dataset_with_index(dataset, sampling_factor=10000):
    """Reads a time series with an index (e.g. time) and resamples.

    Parameters
    ----------
    dataset : String
        File location.
    sampling_factor :
        The time series is sampled down to roughly this number of points by skipping
        every other point.

    Returns
    -------
    Tuple
        data : pd.Series
            The time series (z-score applied) with the index.
        gt : pd:series
            Ground-truth, if available as `dataset`_gt file

    """
    full_path = '../datasets/ground_truth/' + dataset
    data = pd.read_csv(full_path, index_col=0, squeeze=True)
    print("Dataset Original Length n: ", len(data))

    data, factor = _resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    data[:] = zscore(data)

    gt = read_ground_truth(dataset)
    if gt is not None:
        if factor > 1:
            for column in gt:
                gt[column] = gt[column].transform(
                    lambda l: (np.array(l)) // factor)
        return data, gt
    else:
        return data


def pd_series_to_numpy(data):
    """Converts a PD.Series to two numpy arrays by extracting the raw data and index.

    Parameters
    ----------
    data : array or PD.Series
        the TS

    Returns
    -------
    Tuple
        data_index : array_like
            The index of the time series
        data_raw :
            The raw data of the time series

    """
    if isinstance(data, pd.Series):
        data_raw = data.values
        data_index = data.index
    elif isinstance(data, pd.DataFrame):
        data_raw = data.values
        data_index = data.columns
    else:
        data_raw = data
        data_index = np.arange(len(data))
    return data_index, data_raw.astype(np.float64, copy=False)


def read_dataset(dataset, sampling_factor=10000):
    """ Reads a dataset and resamples.

    Parameters
    ----------
    dataset : String
        File location.
    sampling_factor :
        The time series is sampled down to roughly this number of points by skipping
        every other point.

    Returns
    -------
    data : array-like
        The time series with z-score applied.

    """
    full_path = '../datasets/' + dataset
    data = pd.read_csv(full_path).T
    data = np.array(data)[0]
    print("Dataset Original Length n: ", len(data))

    data, factor = _resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    return zscore(data)


@njit(fastmath=True, cache=True)
def _sliding_dot_product(query, time_series):
    """Compute a sliding dot-product using the Fourier-Transform

    Parameters
    ----------
    query : array-like
        first time series, typically shorter than ts
    ts : array-like
        second time series, typically longer than query.

    Returns
    -------
    dot_product : array-like
        The result of the sliding dot-product
    """

    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1

    query = query[::-1]

    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))

    trim = m - 1 + time_series_add

    with objmode(dot_product="float64[:]"):
        dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))

    return dot_product[trim:]


@njit(fastmath=True, cache=True)
def _sliding_mean_std(ts, m):
    """Computes the incremental mean, std, given a time series and windows of length m.

    Computes a total of n-m+1 sliding mean and std-values.

    This implementation is efficient and in O(n), given TS length n.

    Parameters
    ----------
    ts : array-like
        The time series
    m : int
        The length of the sliding window to compute std and mean over.

    Returns
    -------
    Tuple
        movmean : array-like
            The n-m+1 mean values
        movstd : array-like
            The n-m+1 std values
    """
    # if isinstance(ts, pd.Series):
    #     ts = ts.to_numpy()
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ts ** 2)))
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]

    movmean = segSum / m

    # avoid dividing by too small std, like 0
    movstd = np.sqrt(np.clip(segSumSq / m - (segSum / m) ** 2, 0, None))
    movstd = np.where(np.abs(movstd) < 0.1, 1, movstd)

    return [movmean, movstd]


@njit(fastmath=True, cache=True, parallel=True)
def compute_distances_with_knns(ts,
                                m,
                                k,
                                exclude_trivial_match=True,
                                n_jobs=4,
                                slack=0.5):
    """Compute the full Distance Matrix between all pairs of subsequences.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted from
        the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        ts : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool
            Trivial matches will be excluded if this parameter is set
        n_jobs : int
            Number of jobs to be used.
        slack: float
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """
    n = np.int32(ts.shape[0] - m + 1)
    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    D = np.zeros((n, n), dtype=np.float32)
    knns = np.zeros((n, k), dtype=np.int32)

    means, stds = _sliding_mean_std(ts, m)

    dot_first = _sliding_dot_product(ts[:m], ts)
    bin_size = ts.shape[0] // n_jobs
    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, ts.shape[0] - m + 1)

        dot_prev = None
        for order in np.arange(start, end):
            if order == start:
                # O(n log n) operation
                dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
            else:
                # constant time O(1) operations
                dot_rolled = np.roll(dot_prev, 1) \
                             + ts[order + m - 1] * ts[m - 1:n + m] \
                             - ts[order - 1] * np.roll(ts[:n], 1)
                dot_rolled[0] = dot_first[order]

            D[order, :] = distance(dot_rolled, n, m, means, stds, order, halve_m)
            dot_prev = dot_rolled

    for order in prange(D.shape[-1]):
        knn = _argknn(D[order], k, m, n, slack=slack)
        knns[order, :len(knn)] = knn
        knns[order, len(knn):] = -1

    return D, knns


@njit(fastmath=True, cache=True, parallel=True)
def compute_distances_with_knns_sparse(ts,
                                       m,
                                       k,
                                       exclude_trivial_match=True,
                                       n_jobs=4,
                                       slack=0.5):
    """ Compute the full Distance Matrix between all pairs of subsequences of a
        multivariate time series.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted
        from the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        time_series : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool
            Trivial matches will be excluded if this parameter is set
        n_jobs : int
            Number of jobs to be used.
        slack: float
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """
    n = np.int32(ts.shape[0] - m + 1)
    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    D_knn = np.zeros((n, k), dtype=np.float32)
    knns = np.zeros((n, k), dtype=np.int32)

    # TODO: no sparse matrix support in numba. Thus we use this hack
    D_bool = [Dict.empty(key_type=types.int32, value_type=types.bool_) for _ in
              range(n)]

    D_sparse = List()
    for i in range(n):
        D_sparse.append(Dict.empty(key_type=types.int32, value_type=types.float32))

    means, stds = _sliding_mean_std(ts, m)
    dot_first = _sliding_dot_product(ts[:m], ts)
    bin_size = ts.shape[0] // n_jobs

    # first pass, computing the k-nns
    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, n)

        dot_prev = None
        for order in np.arange(start, end):
            if order == start:
                # O(n log n) operation
                dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
            else:
                # constant time O(1) operations
                dot_rolled = np.roll(dot_prev, 1) \
                             + ts[order + m - 1] * ts[m - 1:n + m] \
                             - ts[order - 1] * np.roll(ts[:n], 1)
                dot_rolled[0] = dot_first[order]

            dist = distance(dot_rolled, n, m, means, stds, order, halve_m)
            dot_prev = dot_rolled

            knn = _argknn(dist, k, m, n, slack=slack)
            D_knn[order] = dist[knn]
            knns[order] = knn

    # FIXME: Parallelizm does not work, as Dict is not thread safe :(
    for order in np.arange(0, n):
        # memorize which pairs are needed
        for ks, dist in zip(knns[order], D_knn[order]):
            D_bool[order][ks] = True
            for ks2 in knns[order]:
                D_bool[ks][ks2] = True

    # second pass, filling only the pairs needed
    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, ts.shape[0] - m + 1)

        dot_prev = None
        for order in np.arange(start, end):
            if order == start:
                # O(n log n) operation
                dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
            else:
                # constant time O(1) operations
                dot_rolled = np.roll(dot_prev, 1) \
                             + ts[order + m - 1] * ts[m - 1:n + m] \
                             - ts[order - 1] * np.roll(ts[:n], 1)
                dot_rolled[0] = dot_first[order]

            dist = distance(dot_rolled, n, m, means, stds, order, halve_m)
            dot_prev = dot_rolled

            # fill the knns now with the distances computed
            for key in D_bool[order]:
                D_sparse[order][key] = dist[key]

    return D_sparse, knns


@njit(fastmath=True, cache=True)
def distance(dot_rolled, n, m, means, stds, order, halve_m):
    # Implementation of z-normalized Euclidean distance
    dist = 2 * m * (1 - (dot_rolled - m * means * means[order]) / (
            m * stds * stds[order]))

    # self-join: exclusion zone
    trivialMatchRange = (max(0, order - halve_m),
                         min(order + halve_m, n))
    dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    # allow subsequence itself to be in result
    dist[order] = 0

    return dist


@njit(fastmath=True, cache=True)
def get_radius(D_full, motifset_pos):
    """Computes the radius of the passed motif set (motiflet).

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets

    Returns
    -------
    motiflet_radius : float
        The radius of the motif set
    """
    motiflet_radius = np.inf

    for ii in range(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        current = np.float32(0.0)
        for jj in range(1, len(motifset_pos)):
            if (i != jj):
                j = motifset_pos[jj]
                current = max(current, D_full[i, j])
        motiflet_radius = min(current, motiflet_radius)

    return motiflet_radius


@njit(fastmath=True, cache=True)
def get_pairwise_extent(D_full, motifset_pos, upperbound=np.inf):
    """Computes the extent of the motifset.

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets
    upperbound : float, default: np.inf
        Upper bound on the distances. If passed, will apply admissible pruning
        on distance computations, and only return the actual extent, if it is lower
        than `upperbound`

    Returns
    -------
    motifset_extent : float
        The extent of the motif set, if smaller than `upperbound`, else np.inf
    """

    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)
    for ii in range(len(motifset_pos) - 1):
        i = motifset_pos[ii]

        for jj in range(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]

            motifset_extent = max(motifset_extent, D_full[i][j])
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(fastmath=True, cache=True)
def _argknn(
        dist, k, m, n, lowest_dist=np.inf, slack=0.5):
    """Finds the closest k-NN non-overlapping subsequences in candidates.

    Parameters
    ----------
    dist : array-like
        the distances
    k : int
        The k in k-NN
    m : int
        The window-length
    n : int
        time series length
    lowest_dist : float
        Used for admissible pruning
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    halve_m = int(m * slack)

    dists = np.copy(dist)
    idx = []  # there may be less than k, thus use a list
    for i in range(k):
        pos = np.int32(np.argmin(dists))
        if (not np.isnan(dists[pos])) \
                and (not np.isinf(dists[pos])) \
                and (dists[pos] <= lowest_dist):
            idx.append(pos)

            # exclude all trivial matches
            dists[max(0, pos - halve_m): min(pos + halve_m, n)] = np.inf
        else:
            break
    return np.array(idx, dtype=np.int32)


@njit(fastmath=True, cache=True)
def get_approximate_k_motiflet(
        ts, m, k, D, knns, upper_bound=np.inf
):
    """Compute the approximate k-Motiflets.

    Details are given within the paper Section 4.2 Approximate k-Motiflet Algorithm.

    Parameters
    ----------
    ts : array-like
        The raw time seres
    m : int
        The motif length
    k : int
        The k in k-Motiflets
    D : 2d array-like
        The distance matrix
    upper_bound : float
        Used for admissible pruning

    Returns
    -------
    Tuple
        motiflet_candidate : np.array
            The (approximate) best motiflet found
        motiflet_dist:
            The extent of the motiflet found
    """
    n = len(ts) - m + 1
    motiflet_dist = upper_bound
    motiflet_candidate = None

    motiflet_all_candidates = np.zeros((n, k), dtype=np.int32)

    # allow subsequence itself
    # Fill diagonal with 0
    for i in range(len(D)):
        D[i][i] = 0

    # order by increasing k-nn distance
    knn_distances = np.zeros(n, dtype=np.float32)
    for i in np.arange(n):
        knn_distances[i] = D[i][knns[i, k - 1]]
    best_order = np.argsort(knn_distances)

    # TODO: parallelize??
    for i, order in enumerate(best_order):
        idx = knns[order]

        motiflet_all_candidates[i, :min(k, len(idx))] = idx[:k]
        motiflet_all_candidates[i, len(idx):] = -1

        if len(idx) >= k and idx[k - 1] >= 0:
            if D[order][idx[k - 1]] <= motiflet_dist:
                # get_pairwise_extent() requires the full distance matrix
                motiflet_extent = get_pairwise_extent(D, idx[:k], motiflet_dist)
                if motiflet_extent <= motiflet_dist:
                    motiflet_dist = motiflet_extent
                    motiflet_candidate = idx[:k]
            else:
                break

    return motiflet_candidate, motiflet_dist, motiflet_all_candidates


@njit(fastmath=True, cache=True)
def _check_unique(motifset_1, motifset_2, motif_length):
    """Check for overlaps between two motif sets.

    Two motif sets overlapp, if more than m/2 subsequences overlap from motifset 1.

    Parameters
    ----------
    motifset_1 : array-like
        Positions of the smaller motif set.
    motifset_2 : array-like
        Positions of the larger motif set.
    motif_length : int
        The length of the motif. Overlap exists, if 25% of two subsequences overlap.

    Returns
    -------
    True, if there are at least m/2 subsequences with an overlap of 25%, else False.
    """
    count = 0
    for a in motifset_1:  # smaller motiflet
        for b in motifset_2:  # larger motiflet
            if abs(a - b) < (motif_length / 4):
                count = count + 1
                break

        if count >= len(motifset_1) / 2:
            return False
    return True


# @njit(fastmath=True, cache=True)
def _filter_unique(elbow_points, candidates, motif_length):
    """Filters the list of candidate elbows for only the non-overlapping motifsets.

    This method applied a duplicate detection by filtering overlapping motif sets.
    Two candidate motif sets overlap, if at least m/2 subsequences of the smaller
    motifset overlapp with the larger motifset. Only the largest non-overlapping
    motif sets are retained.

    Parameters
    ----------
    elbow_points : array-like
        List of possible k's for elbow-points.
    candidates : 2d array-like
        List of motif sets for each k
    motif_length : int
        Length of the motifs, needed for checking overlaps.

    Returns
    -------
    filtered_ebp : array-like
        The set of non-overlapping elbow points.

    """
    filtered_ebp = []
    for i in range(len(elbow_points)):
        unique = True
        for j in range(i + 1, len(elbow_points)):
            unique = _check_unique(
                candidates[elbow_points[i]], candidates[elbow_points[j]], motif_length)
            if not unique:
                break
        if unique:
            filtered_ebp.append(elbow_points[i])

    return np.array(filtered_ebp)


@njit(fastmath=True, cache=True)
def find_elbow_points(dists, alpha=2, elbow_deviation=1.00):
    """Finds elbow-points in the elbow-plot (extent over each k).

    Parameters
    ----------
    dists : array-like
        The extends for each k.
    alpha : float
        A threshold used to detect an elbow-point in the distances.
        It measures the relative change in deviation from k-1 to k to k+1.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    elbow_points : the elbow-points in the extent-function
    """
    elbow_points = set()
    elbow_points.add(2)  # required for numba to have a type
    elbow_points.clear()

    peaks = np.zeros(len(dists))
    for i in range(3, len(peaks) - 1):
        if (dists[i] != np.inf and
                dists[i + 1] != np.inf and
                dists[i - 1] != np.inf):

            m1 = (dists[i + 1] - dists[i]) + 0.00001
            m2 = (dists[i] - dists[i - 1]) + 0.00001

            # avoid detecting elbows in near constant data
            # TODO adding this removes reproducability
            # if dists[i - 1] == dists[i]:
            #    m2 = 1.0  # peaks[i] = 0

            if (dists[i] > 0) and (dists[i + 1] / dists[i] > elbow_deviation):
                peaks[i] = (m1 / m2)

    elbow_points = []

    while True:
        p = np.argmax(peaks)
        if peaks[p] > alpha:
            elbow_points.append(p)
            peaks[p - 1:p + 2] = 0
        else:
            break

    if len(elbow_points) == 0:
        elbow_points.append(2)

    return np.sort(np.array(list(set(elbow_points))))


def find_au_ef_motif_length(
        data,
        k_max,
        motif_length_range,
        exclusion=None,
        elbow_deviation=1.00,
        slack=0.5,
        subsample=2):
    """Computes the Area under the Elbow-Function within an of motif lengths.

    Parameters
    ----------
    data : array-like
        The time series.
    k_max : int
        The interval of k's to compute the area of a single AU_EF.
    motif_length_range : array-like
        The range of lengths to compute the AU-EF.
    exclusion : 2d-array
        exclusion zone - use when searching for the TOP-2 motiflets
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

    Returns
    -------
    Tuple
        minimum : array-like
            The minumum found
        all_minima : array-like
            All local minima found
        au_efs : array-like
            For each length in the interval, the AU_EF.
        elbows :
            Largest k (largest elbow) found
        top_motiflets :
            The motiflet for the largest k for each length.

    """
    # apply sampling for speedup only
    if subsample > 1:
        data = data[::subsample]

    # in reverse order
    au_efs = np.zeros(len(motif_length_range), dtype=object)
    au_efs.fill(np.inf)
    elbows = np.zeros(len(motif_length_range), dtype=object)
    top_motiflets = np.zeros(len(motif_length_range), dtype=object)
    dists = np.zeros(len(motif_length_range), dtype=object)

    # stores the position of the l+1 motif set as an approximate pos for l
    approximate_pos = None

    # TODO parallelize?
    for i, m in enumerate(motif_length_range[::-1]):
        if m // subsample < data.shape[0]:
            dist, candidates, elbow_points, _ = search_k_motiflets_elbow(
                k_max,
                data,
                m // subsample,
                exclusion=exclusion,
                # TODO this can cause an error with SLACK set?
                approximate_motiflet_pos=approximate_pos,
                elbow_deviation=elbow_deviation,
                slack=slack)

            dists_ = dist[(~np.isinf(dist)) & (~np.isnan(dist))]
            if dists_.max() - dists_.min() == 0:
                au_efs[i] = 1.0
            else:
                au_efs[i] = (((dists_ - dists_.min()) / (
                        dists_.max() - dists_.min())).sum()
                             / len(dists_))

            elbow_points = _filter_unique(elbow_points, candidates, m // subsample)

            top_motiflet = None
            if len(elbow_points > 0):
                elbows[i] = elbow_points
                top_motiflets[i] = candidates[elbow_points]
            else:
                # we found only the pair motif
                elbows[i] = [2]
                top_motiflets[i] = [candidates[2]]

                # no elbow can be found, ignore this part
                au_efs[i] = 1.0

            dists[i] = dist
            approximate_pos = candidates

    # reverse order
    au_efs = np.array(au_efs, dtype=np.float64)[::-1]
    elbows = elbows[::-1]
    dists = dists[::-1]
    top_motiflets = top_motiflets[::-1] * subsample

    # Minima in AU_EF
    minimum = motif_length_range[np.nanargmin(au_efs)]
    au_ef_minima = argrelextrema(au_efs, np.less_equal, order=subsample)

    # Maxima in the EF
    return (minimum, au_ef_minima, au_efs,
            elbows, top_motiflets, dists)


def search_k_motiflets_elbow(
        k_max,
        data,
        motif_length='auto',
        motif_length_range=None,
        exclusion=None,
        approximate_motiflet_pos=None,
        elbow_deviation=1.00,
        filter=True,
        slack=0.5
):
    """Computes the elbow-function.

    This is the method to find the characteristic k-Motiflets within range
    [2...k_max] for given a `motif_length` using elbow-plots.

    Details are given within the paper Section 5.1 Learning meaningful k.

    Parameters
    ----------
    k_max : int
        use [2...k_max] to compute the elbow plot (user parameter).
    data : array-like
        the TS
    motif_length : int
        the length of the motif (user parameter) or
        `motif_length == 'AU_EF'` or `motif_length == 'auto'`.
    motif_length_range : array-like
        Can be used to determine to length of the motif set automatically.
        If a range is passed and `motif_length == 'auto'`, the best window length
        is first determined, prior to computing the elbow-plot.
    exclusion : 2d-array
        exclusion zone - use when searching for the TOP-2 motiflets
    approximate_motiflet_pos : array-like
        An initial estimate of the positions of the k-Motiflets for each k in the
        given range [2...k_max]. Will be used for bounding distance computations.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    filter: bool, default=True
        filters overlapping motiflets from the result,
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.


    Returns
    -------
    Tuple
        dists :
            distances for each k in [2...k_max]
        candidates :
            motifset-candidates for each k
        elbow_points :
            elbow-points
        m : int
            best motif length
    """
    # convert to numpy array
    _, data_raw = pd_series_to_numpy(data)

    # auto motif size selection
    if motif_length == 'AU_EF' or motif_length == 'auto':
        if motif_length_range is None:
            print("Warning: no valid motiflet range set")
            assert False
        m, _, _, _ = find_au_ef_motif_length(
            data, k_max, motif_length_range,
            elbow_deviation=elbow_deviation,
            slack=slack)
    elif isinstance(motif_length, int) or \
            isinstance(motif_length, np.int32) or \
            isinstance(motif_length, np.int64):
        m = motif_length
    else:
        print("Warning: no valid motif_length set - use 'auto' for automatic selection")
        assert False

    # non-overlapping motifs only
    n = data_raw.shape[0] - m + 1
    k_max_ = max(3, min(int(n // (m * slack)), k_max))

    # non-overlapping motifs only
    k_motiflet_distances = np.zeros(k_max_)
    k_motiflet_candidates = np.empty(k_max_, dtype=object)

    # switch to sparse matrix representation when length is above 30_000
    # sparse matrix is 2x slower but needs less memory
    sparse = n >= 30000
    if not sparse:
        D_full, knns = compute_distances_with_knns(data_raw, m, k_max_, slack=slack)
    else:
        D_full, knns = compute_distances_with_knns_sparse(data_raw, m, k_max_, slack=slack)

    exclusion_m = int(m * slack)

    upper_bound = np.inf
    for test_k in tqdm(range(k_max_ - 1, 1, -1),
                       desc='Compute ks (' + str(k_max_) + ")",
                       position=0, leave=False):

        # Top-N retrieval
        if exclusion is not None and exclusion[test_k] is not None:
            if not sparse:
                for pos in exclusion[test_k].flatten():
                    if pos is not None:
                        trivialMatchRange = (max(0, pos - exclusion_m),
                                             min(pos + exclusion_m, len(D_full)))
                    D_full[:, trivialMatchRange[0]:trivialMatchRange[1]] = np.inf
            else:
                raise Exception('Top-k is not supported for sparse matrices.')

        # Does not work
        # use an approximate position as an initial estimate, if available
        # bound_set = False
        # if approximate_motiflet_pos is not None \
        #         and len(approximate_motiflet_pos) > test_k \
        #         and approximate_motiflet_pos[test_k] is not None:
        #     dd = get_pairwise_extent(D_full, approximate_motiflet_pos[test_k])
        #     upper_bound = min(dd, upper_bound)
        #     bound_set = True

        candidate, candidate_dist, _ = get_approximate_k_motiflet(
            data_raw, m, test_k, D_full, knns,
            upper_bound=upper_bound,
        )

        # if candidate is None and bound_set:
        #     # If we already found the best motif in length l+1
        #     candidate = approximate_motiflet_pos[test_k]
        #     candidate_dist = dd

        k_motiflet_distances[test_k] = candidate_dist
        k_motiflet_candidates[test_k] = candidate
        upper_bound = min(candidate_dist, upper_bound)

        # compute a new upper bound
        if candidate is not None:
            dist_new = get_pairwise_extent(D_full, candidate[:test_k])
            upper_bound = min(upper_bound, dist_new)

    # smoothen the line to make it monotonically increasing
    k_motiflet_distances[0:2] = k_motiflet_distances[2]
    for i in range(len(k_motiflet_distances), 2):
        k_motiflet_distances[i - 1] = min(k_motiflet_distances[i],
                                          k_motiflet_distances[i - 1])

    elbow_points = find_elbow_points(k_motiflet_distances,
                                     elbow_deviation=elbow_deviation)

    if filter:
        elbow_points = _filter_unique(
            elbow_points, k_motiflet_candidates, motif_length)

    return k_motiflet_distances, k_motiflet_candidates, elbow_points, m


@njit(fastmath=True, cache=True)
def candidate_dist(D_full, pool, upperbound, m, slack=0.5):
    motiflet_candidate_dist = 0
    m_half = int(m * slack)
    for i in pool:
        for j in pool:
            if ((i != j and np.abs(i - j) < m_half)
                    or (i != j and D_full[i, j] > upperbound)):
                return np.inf

    for i in pool:
        for j in pool:
            motiflet_candidate_dist = max(motiflet_candidate_dist, D_full[i, j])

    return motiflet_candidate_dist


@njit(fastmath=True, cache=True)
def find_k_motiflets(ts, D_full, m, k, upperbound=None, slack=0.5):
    """Exact algorithm to compute k-Motiflets

    Warning: The algorithm has exponential runtime complexity.

    Parameters
    ----------
    ts : array-like
        The time series
    D_full : 2d array-like
        The pairwise distance matrix
    m : int
        Length of the motif
    k : int
        k-Motiflet size
    upperbound : float
        Admissible pruning on distance computations.

    Returns
    -------
    best found motiflet and its extent.
    """
    n = len(ts) - m + 1

    motiflet_dist = upperbound
    if upperbound is None:
        motiflet_candidate, motiflet_dist, _ = get_approximate_k_motiflet(
            ts, m, k, D_full, upper_bound=np.inf, slack=slack)

        motiflet_pos = motiflet_candidate

    # allow subsequence itself
    np.fill_diagonal(D_full, 0)
    k_halve_m = k * int(m * slack)

    def exact_inner(ii, k_halve_m, D_full,
                    motiflet_dist, motiflet_pos, m):

        for i in np.arange(ii, min(n, ii + m)):  # in runs of m
            D_candidates = np.argwhere(D_full[i] <= motiflet_dist).flatten()
            if (len(D_candidates) >= k and
                    np.ptp(D_candidates) > k_halve_m):
                # exhaustive search over all subsets
                for permutation in itertools.combinations(D_candidates, k):
                    if np.ptp(permutation) > k_halve_m:
                        dist = candidate_dist(D_full, permutation, motiflet_dist, m,
                                              slack)
                        if dist < motiflet_dist:
                            motiflet_dist = dist
                            motiflet_pos = np.copy(permutation)
        return motiflet_dist, motiflet_pos

    motiflet_dists, motiflet_poss = zip(*Parallel(n_jobs=-1)(
        delayed(exact_inner)(
            i,
            k_halve_m,
            D_full,
            motiflet_dist,
            motiflet_pos,
            m
        ) for i in range(0, n, m)))

    min_pos = np.nanargmin(motiflet_dists)
    motiflet_dist = motiflet_dists[min_pos]
    motiflet_pos = motiflet_poss[min_pos]

    return motiflet_dist, motiflet_pos


@njit(fastmath=True, cache=True, parallel=True)
def compute_distances_full(ts,
                           m,
                           exclude_trivial_match=True,
                           n_jobs=4,
                           slack=0.5):
    """Compute the full Distance Matrix between all pairs of subsequences.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted from
        the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        ts : array-like
            The time series
        m : int
            The window length
        exclude_trivial_match : bool
            Trivial matches will be excluded if this parameter is set
        n_jobs : int
            Number of jobs to be used.
        slack: float
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """

    D, _ = compute_distances_with_knns(ts, m, k=1,
                                       exclude_trivial_match=exclude_trivial_match,
                                       n_jobs=n_jobs,
                                       slack=slack)
    return D
