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
from numba import njit, prange, objmode, typed
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
def compute_distances_full(ts, m, exclude_trivial_match=True, n_jobs=4, slack=0.5):
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
            Number of jobs to used

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences

    """
    n = np.int32(ts.shape[0] - m + 1)
    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    D = np.zeros((n, n), dtype=np.float32)
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

    return D


@njit(fastmath=True, cache=True)
def distance(dot_rolled, n, m, means, stds, order, halve_m):
    # there is a numba bug, thus we have to repeat all codes:
    # https: // github.com / numba / numba / issues / 7681
    dist = 2 * m * (1 - (dot_rolled - m * means * means[order]) / (
            m * stds * stds[order]))

    # self-join: exclusion zone
    trivialMatchRange = (max(0, order - halve_m),
                         min(order + halve_m, n))
    dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    # allow subsequence itself to be in result
    dist[order] = 0

    return dist


def compute_distances_full_seq(ts, m, exclude_trivial_match=True, slack=0.5):
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

    Returns
    -------
    D : 2d array-like
        The O(n^2) z-normed ED distances between all pairs of subsequences

    """
    n = len(ts) - m + 1
    halve_m = 0
    if exclude_trivial_match:
        halve_m = int(m * slack)

    D = np.zeros((n, n), dtype=np.float32)
    dot_prev = None
    means, stds = _sliding_mean_std(ts, m)

    for order in range(0, n):

        # first iteration O(n log n)
        if order == 0:
            dot_first = _sliding_dot_product(ts[:m], ts)
            dot_rolled = dot_first
        # O(1) further operations
        else:
            dot_rolled = np.roll(dot_prev, 1) + ts[order + m - 1] * ts[m - 1:n + m] - \
                         ts[order - 1] * np.roll(ts[:n], 1)
            dot_rolled[0] = dot_first[order]

        x_mean = means[order]
        x_std = stds[order]

        dist = 2 * m * (1 - (dot_rolled - m * means * x_mean) / (m * stds * x_std))

        # self-join: eclusion zone
        trivialMatchRange = (max(0, order - halve_m),
                             min(order + halve_m, n))
        dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

        # allow subsequence itself to be in result
        dist[order] = 0
        D[order, :] = dist

        dot_prev = dot_rolled

    return D


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

            motifset_extent = max(motifset_extent, D_full[i, j])
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(fastmath=True, cache=True)
def _get_top_k_non_trivial_matches_inner(
        dist, k, candidates, lowest_dist=np.inf):
    """Filters a list of potential non-overlapping k'-NNs for the closest k ones.

    Parameters
    ----------
    dist : array-like
        the distances
    k : int
        The k in k-NN
    candidates:
        The list of k'>k potential candidate subsequences, must be non-overlapping
    lowest_dist:
        The best known lowest_dist. Only those subsequences lower than `lowest_dist`
        are returned

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    # admissible pruning: are there enough offsets within range?
    p = 0
    for i in range(len(candidates), 0, -1):
        if dist[candidates[i - 1]] <= lowest_dist:
            p = i
            break
    return candidates[:min(k, p)]


@njit(fastmath=True, cache=True)
def _get_top_k_non_trivial_matches(
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

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    dist_idx = np.argwhere(dist <= lowest_dist).flatten().astype(np.int32)
    halve_m = int(m * slack)

    dists = np.copy(dist)
    idx = []  # there may be less than k, thus use a list
    for i in range(k):
        pos = dist_idx[np.argmin(dists[dist_idx])]
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
def _get_top_k_non_trivial_matches_new(
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

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    dist_idx = np.argwhere(dist <= lowest_dist).flatten().astype(np.int32)
    halve_m = int(m * slack)

    idx = []  # there may be less than k, thus use a list
    for i in range(k):
        current_min = lowest_dist
        current_pos = -1

        for pos in dist_idx:
            if dist[pos] <= current_min \
                    and (not np.isnan(dist[pos])) \
                    and (not np.isinf(dist[pos])):
                overlap = False
                for ks in idx:
                    # exclude all trivial matches
                    if max(-1, ks - halve_m) < pos < min(ks + halve_m, n):
                        overlap = True
                        break

                if not overlap:
                    current_min = dist[pos]
                    current_pos = pos

        if current_pos >= 0:
            idx.append(current_pos)
        else:  # nothing left
            break

    return np.array(idx, dtype=np.int32)


@njit(fastmath=True, cache=True)
def get_approximate_k_motiflet(
        ts, m, k, D,
        upper_bound=np.inf,
        incremental=False,
        all_candidates=np.zeros((1, 1), dtype=np.int32),  # Empty type to fool numba
        slack=0.5
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
    incremental : bool, default: False
        When set to True, must also provide `all_candidates`
    all_candidates : 2d array-like
        We can reduce a set of k'-Motiflets, with k'>k, to a k-Motiflet. Used for
        efficient computation of elbows from large to small.

    Returns
    -------
    Tuple
        motiflet_candidate : np.array
            The (approximate) best motiflet found
        motiflet_dist:
            The extent of the motiflet found
        motiflet_all_candidates : 2d array-like
            For each subsequence, a motifset, with minimal extent, found containing it.
            Used for refinement in incremental computation `incremental=True`.
    """
    n = len(ts) - m + 1
    motiflet_dist = upper_bound
    motiflet_candidate = None

    motiflet_all_candidates = np.zeros((n, k), dtype=np.int32)

    # allow subsequence itself
    np.fill_diagonal(D, 0)

    # TODO: parallelize??
    for i, order in enumerate(np.arange(n)):
        dist = D[order]

        if incremental:
            idx = _get_top_k_non_trivial_matches_inner(
                dist, k, all_candidates[order], motiflet_dist)
        else:
            idx = _get_top_k_non_trivial_matches(dist, k, m, n, motiflet_dist, slack)

        motiflet_all_candidates[i, :len(idx)] = idx
        motiflet_all_candidates[i, len(idx):] = -1

        if len(idx) >= k and dist[idx[-1]] <= motiflet_dist:
            # get_pairwise_extent requires the full matrix
            motiflet_extent = get_pairwise_extent(D, idx[:k], motiflet_dist)
            if motiflet_extent <= motiflet_dist:
                motiflet_dist = motiflet_extent
                motiflet_candidate = idx[:k]

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
            filtered_ebp.append(elbow_points[i], )

    # print("Elbows", filtered_ebp)
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
            # TODO need to test this?
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


def _inner_au_ef(data, k_max, m,
                 approximate_motiflet_pos=None,
                 elbow_deviation=1.00,
                 slack=0.5):
    """Computes the Area under the Elbow-Function within an interval [2...k_max].

    Parameters
    ----------
    data : array-like
        The raw time series data.
    k_max : int
        Largest k. All k's within [2...k_max] are computed.
    m : int
        Motif length
    approximate_motiflet_pos : array-like
        An initial estimate of the positions of the k-Motiflets for each k in the
        given range [2...k_max]. Will be used for bounding distance computations.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    Tuple
        au_efs : float
            Area under the EF
        elbows : array-like
            Elbows found
        top_motiflet:
            Largest motiflet found (largest k), given the elbows.
        dists : array-like
            Distances for each k in the given interval

    """
    dists, candidates, elbow_points, _ = search_k_motiflets_elbow(
        k_max,
        data,
        m,
        approximate_motiflet_pos=approximate_motiflet_pos,
        elbow_deviation=elbow_deviation,
        slack=slack)

    dists = dists[(~np.isinf(dists)) & (~np.isnan(dists))]
    au_efs = ((dists - dists.min()) / (dists.max() - dists.min())).sum() / len(dists)
    elbow_points = _filter_unique(elbow_points, candidates, m)

    top_motiflet = None
    if len(elbow_points > 0):
        elbows = len(elbow_points)
        top_motiflet = candidates[elbow_points[-1]]
    else:
        # pair motif
        elbows = 1
        top_motiflet = candidates[0]

    return au_efs, elbows, top_motiflet, dists, candidates


def find_au_ef_motif_length(
        data,
        k_max,
        motif_length_range,
        elbow_deviation=1.00,
        slack=0.5):
    """Computes the Area under the Elbow-Function within an of motif lengths.

    Parameters
    ----------
    data : array-like
        The time series.
    k_max : int
        The interval of k's to compute the area of a single AU_EF.
    motif_length_range : array-like
        The range of lengths to compute the AU-EF.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    Tuple
        length : array-like
            The range of lengths searched.
        au_efs : array-like
            For each length in the interval, the AU_EF.
        elbows :
            The largest k found for each length.
        top_motiflets :
            The motiflet for the largest k for each length.

    """
    # apply sampling for speedup only
    subsample = 2
    data = data[::subsample]

    # index = (data.index / subsample) if isinstance(data, pd.Series) else np.arange(
    #     len(data))

    # in reverse order
    au_efs = np.zeros(len(motif_length_range), dtype=object)
    au_efs.fill(np.inf)
    elbows = np.zeros(len(motif_length_range), dtype=object)
    top_motiflets = np.zeros(len(motif_length_range), dtype=object)

    # stores the position of the l+1 motif set as an approximate pos for l
    approximate_pos = None

    # TODO parallelize?
    for i, m in enumerate(motif_length_range[::-1]):
        if m < data.shape[0]:
            au_efs[i], elbows[i], top_motiflets[i], _, approximate_pos \
                = _inner_au_ef(
                        data, k_max, int(m / subsample),
                        approximate_motiflet_pos=approximate_pos,
                        elbow_deviation=elbow_deviation,
                        slack=slack)

    au_efs = np.array(au_efs, dtype=np.float64)[::-1]
    elbows = elbows[::-1]
    top_motiflets = top_motiflets[::-1]

    # if no elbow can be found, ignore this part
    condition = np.argwhere(elbows == 0).flatten()
    au_efs[condition] = np.inf

    return motif_length_range[np.nanargmin(au_efs)], au_efs, elbows, top_motiflets


def search_k_motiflets_elbow(
        k_max,
        data,
        motif_length='auto',
        motif_length_range=None,
        exclusion=None,
        approximate_motiflet_pos=None,
        elbow_deviation=1.00,
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
    k_max_ = max(3, min(int(len(data) / (m * slack)), k_max))

    k_motiflet_distances = np.zeros(k_max_)
    k_motiflet_candidates = np.empty(k_max_, dtype=object)

    D_full = compute_distances_full(data_raw, m, slack)

    exclusion_m = int(m * slack)
    motiflet_candidates = np.zeros((D_full.shape[0], 1), dtype=np.int32)

    upper_bound = np.inf
    incremental = False
    for test_k in tqdm(range(k_max_ - 1, 1, -1),
                       desc='Compute ks (' + str(k_max_) + ")",
                       position=0, leave=False):
        # Top-N retrieval
        if exclusion is not None and exclusion[test_k] is not None:
            for pos in exclusion[test_k].flatten():
                if pos is not None:
                    trivialMatchRange = (max(0, pos - exclusion_m),
                                         min(pos + exclusion_m, len(D_full)))
                    D_full[:, trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

        # use an approximate position as an initial estimate, if available
        if approximate_motiflet_pos is not None \
                and len(approximate_motiflet_pos) > test_k \
                and approximate_motiflet_pos[test_k] is not None:
            dd = get_pairwise_extent(D_full, approximate_motiflet_pos[test_k])
            upper_bound = min(dd, upper_bound)

        candidate, candidate_dist, all_candidates = get_approximate_k_motiflet(
            data_raw, m, test_k, D_full,
            upper_bound=upper_bound,
            incremental=incremental,  # we use an incremental computation
            all_candidates=motiflet_candidates,
            slack=slack
        )

        if not incremental:
            motiflet_candidates = all_candidates

        incremental = True

        if candidate is None and \
                len(k_motiflet_candidates) > test_k + 1 and \
                k_motiflet_candidates[test_k + 1] is not None:
            # This should not happen, but does?
            candidate = k_motiflet_candidates[test_k + 1][:test_k]
            candidate_dist = get_pairwise_extent(D_full, candidate)

        k_motiflet_distances[test_k] = candidate_dist
        k_motiflet_candidates[test_k] = candidate
        upper_bound = min(candidate_dist, upper_bound)

    # smoothen the line to make it monotonically increasing
    k_motiflet_distances[0:2] = k_motiflet_distances[2]
    for i in range(len(k_motiflet_distances), 2):
        k_motiflet_distances[i - 1] = min(k_motiflet_distances[i],
                                          k_motiflet_distances[i - 1])

    elbow_points = find_elbow_points(k_motiflet_distances,
                                     elbow_deviation=elbow_deviation)
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
