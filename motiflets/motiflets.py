import itertools


import numpy as np
import numpy.fft as fft
import pandas as pd

from os.path import exists
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import zscore
from tqdm import tqdm
from ast import literal_eval


def as_series(data, index_range, index_name):
    series = pd.Series(data=data, index=index_range)
    series.index.name = index_name
    return series


def resample(data, sampling_factor=10000):
    if len(data) > sampling_factor:
        factor = np.int32(len(data) / sampling_factor)
        data = data[::factor]
    return data


def read_ground_truth(dataset):
    file = '../datasets/ground_truth/' + dataset.split(".")[0]+"_gt.csv"
    if (exists(file)):
        print(file)
        series = pd.read_csv(
            file, index_col=0, 
            converters={1: literal_eval, 2: literal_eval,  3: literal_eval})
        return series
    return None


def read_dataset_with_index(dataset, sampling_factor=10000):
    full_path = '../datasets/ground_truth/' + dataset
    data = pd.read_csv(full_path, index_col=0, squeeze=True)
    print("Dataset Original Length n: ", len(data))

    data = resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    data[:] = zscore(data)

    gt = read_ground_truth(dataset)
    if gt is not None:
        return data, gt
    else:
        return data


def read_dataset(dataset, sampling_factor=10000):
    full_path = '../datasets/' + dataset
    data = pd.read_csv(full_path).T
    data = np.array(data)[0]
    print("Dataset Original Length n: ", len(data))

    data = resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))
    
    # gt = read_ground_truth(dataset)    
    return zscore(data)# , gt



def calc_sliding_window(time_series, window):
    shape = time_series.shape[:-1] + (time_series.shape[-1] - window + 1, window)
    strides = time_series.strides + (time_series.strides[-1],)
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)


def sliding_dot_product(query, ts):
    m = len(query)
    n = len(ts)

    ts_add = 0
    if n % 2 == 1:
        ts = np.insert(ts, 0, 0)
        ts_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    query = query[::-1]
    query = np.pad(query, (0, n - m + ts_add - q_add), 'constant')
    trim = m - 1 + ts_add
    dot_product = fft.irfft(fft.rfft(ts) * fft.rfft(query))
    return dot_product[trim:]


def sliding_mean_std(ts, m):
    if isinstance(ts, pd.Series):
        ts = ts.to_numpy()
    s = np.insert(np.cumsum(ts), 0, 0)
    sSq = np.insert(np.cumsum(ts ** 2), 0, 0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]

    movmean = segSum / m
    movstd = np.sqrt(segSumSq / m - (segSum / m) ** 2)

    # avoid dividing by too small std, like 0
    movstd = np.where(abs(movstd) < 0.1, 1, movstd)

    return [movmean, movstd]


# Distance Matrix with Dot-Product / no-loops
def compute_distances_full(ts, m):
    n = len(ts) - m + 1
    halve_m = int(m / 2)

    D = np.zeros((n, n), dtype=np.float32)
    dot_prev = None
    means, stds = sliding_mean_std(ts, m)

    for order in range(0, n):

        # first iteration O(n log n)
        if order == 0:
            dot_first = sliding_dot_product(ts[:m], ts)
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


@njit(fastmath=True)
def get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf):
    """ Requires the full matrix!!! """

    # for i in motiflet_pos:
    #    for j in motiflet_pos[i+1::]:

    motiflet_range = np.float32(0.0)
    for ii in range(len(motiflet_pos) - 1):
        i = motiflet_pos[ii]
        for jj in range(ii + 1, len(motiflet_pos)):
            j = motiflet_pos[jj]

            motiflet_range = max(motiflet_range, D_full[i, j])
            if motiflet_range >= upperbound:
                return np.inf

    return motiflet_range


@njit
def get_top_k_non_trivial_matches(
        dist, k, m, n, order, lowest_dist=np.inf, dists=None):
    halve_m = int(m / 2)

    # admissible pruning: are there enough offsets within range?
    idx2 = np.argwhere(dist < lowest_dist).flatten()
    if (len(idx2) < k or
            # (np.max(idx2) - np.min(idx2) < k * halve_m)
            np.ptp(idx2) < k * halve_m):
        return np.array([order], dtype=np.int32)

    if dists is None:
        dists = np.copy(dist)
    else:  # avoid allocating memory again
        dists[:] = dist

    idx = []  # there may be less than k, thus use a list
    for i in range(k):
        pos = np.argmin(dists)
        if dists[pos] < lowest_dist:
            idx.append(pos)
            dists[max(0, pos - halve_m):min(pos + halve_m, n)] = np.inf
        else:
            break

    return np.array(idx, dtype=np.int32)


@njit
def get_approximate_k_motiflet_inner(
        n, m, k, D, offset, upper_bound=np.inf):
    lowest_dist = upper_bound
    motiflet_candidate = None

    # allow subsequence itself
    np.fill_diagonal(D, 0)

    # avoid allocating memory over an over
    dist_buffer = np.copy(D[0])

    for order in np.arange(n):
        modulo = order % 4
        if modulo == offset:
            dist = np.copy(D[order])
            # dist[order] = 0
            idx = get_top_k_non_trivial_matches(
                dist, k, m, n, order, lowest_dist, dists=dist_buffer)

            if len(idx) >= k and dist[idx[-1]] <= lowest_dist:
                # Get get_pairwise_extent requires the full matrix 
                motiflet_extent = get_pairwise_extent(D, idx[:k], lowest_dist)
                if lowest_dist > motiflet_extent:
                    lowest_dist = motiflet_extent
                    motiflet_candidate = idx[:k]

    return motiflet_candidate, lowest_dist


@njit
def get_approximate_k_motiflet(ts, m, k, D, upper_bound=np.inf):
    n = len(ts) - m + 1

    """
    # is not faster, as admissible pruning does not work :(
    # iterate all subsequences
    result = Parallel(n_jobs=4)(    
        delayed(get_approximate_k_motiflet_inner)(
                n, m, k, D, i, upper_bound=np.inf
            ) for i in np.arange(4))

    result = np.array(result)
    candidates = result[:,0]
    dists = result[:,1]
    motiflet_candidate = candidates[np.argmin(dists)]
    motiflet_dist = dists[np.argmin(dists)]
    """

    motiflet_dist = upper_bound
    motiflet_candidate = None

    for i in range(0, 4):
        candidate, dist = get_approximate_k_motiflet_inner(
            n, m, k, D, i, upper_bound=motiflet_dist)
        if dist < motiflet_dist:
            motiflet_dist = dist
            motiflet_candidate = candidate

    return motiflet_candidate, motiflet_dist


@njit
def find_elbow_points(dists):
    elbow_points = set()
    elbow_points.add(2)
    elbow_points.clear()

    peaks = np.zeros(len(dists))
    for i in range(3, len(peaks) - 1):
        if (dists[i] != np.inf and
                dists[i + 1] != np.inf and
                dists[i - 1] != np.inf):
            m1 = (dists[i + 1] - dists[i]) + 0.00001
            m2 = (dists[i] - dists[i - 1]) + 0.00001
            peaks[i] = m1 / m2

    # elbow_points = [2]
    elbow_points = []

    while True:
        p = np.argmax(peaks)
        if peaks[p] > 2:
            elbow_points.append(p)
            peaks[p - 1:p + 2] = 0
        else:
            break

    return np.sort(np.array(list(set(elbow_points))))


def inner_au_pef(data, dataset, ks, index, m):
    dists, candidates, elbow_points, _ = search_k_motiflets_elbow(
        ks,
        data,
        dataset,
        m)

    if np.isnan(dists).any() or np.isinf(dists).any():
        return None, None, None

    au_pefs = ((dists - dists.min()) / (dists.max() - dists.min())).sum() / len(dists)
    elbow = len(elbow_points)

    top_motiflet = None
    if len(elbow_points > 0):
        top_motiflet = candidates[elbow_points[-1]]

    print("Motif Length:", m, "\t", index[m],
          "\tAU_PEF:", np.round(au_pefs, 3),
          "\t#Elbows:", elbow)

    return au_pefs, elbow, top_motiflet


def find_au_pef_motif_length(data, dataset, ks, motif_length_range):
    subsample = 2
    data = data[::subsample]

    index = (data.index / subsample) if isinstance(data, pd.Series) else np.arange(
        len(data))

    # TODO parallel not possible when elbows are parallel, too
    results = Parallel(n_jobs=1)(delayed(inner_au_pef)(
        data, dataset, ks, index, int(m / subsample)) for i, m in
                                 enumerate(motif_length_range))

    results = np.array(results)
    au_pefs = np.array(results[:, 0], dtype=np.float64)
    elbows = results[:, 1]
    top_motiflets = results[:, 2]

    # if no elbow can be found, ignore this part
    condition = np.argwhere(elbows == 0).flatten()
    au_pefs[condition] = np.inf

    return motif_length_range[np.argmin(au_pefs)], au_pefs, elbows, top_motiflets


def search_k_motiflets_elbow(ks,
                             data,
                             dataset,
                             motif_length='auto',
                             motif_length_range=None,
                             exclusion=None):
    # convert to numpy array
    data_raw = data
    if isinstance(data, pd.Series):
        data_raw = data.to_numpy()

        # auto motif size selection
    if motif_length == 'AU_PEF' or motif_length == 'auto':
        if motif_length_range is None:
            print("Warning: no valid motiflet range set")
            assert False
        m, _, _, _ = find_au_pef_motif_length(
            data, dataset, ks, motif_length_range)
    elif isinstance(motif_length, int) or \
            isinstance(motif_length, np.int32) or \
            isinstance(motif_length, np.int64):
        m = motif_length
    else:
        print("Warning: no valid motif_length set - use 'auto' for automatic selection")
        assert False

    k_motiflet_distances = np.zeros(ks)
    k_motiflet_candidates = np.empty(ks, dtype=object)

    D_full = compute_distances_full(data_raw, m)

    upper_bound = np.inf
    exclusion_m = int(m / 3)

    for test_k in tqdm(range(ks - 1, 1, -1), desc='Compute ks'):
        # for test_k in range(ks - 1, 1, -1):
        # print(".", end='')
        if exclusion is not None and exclusion[test_k] is not None:
            for pos in exclusion[test_k].flatten():
                if pos is not None:
                    trivialMatchRange = (max(0, pos - exclusion_m),
                                         min(pos + exclusion_m, len(D_full)))
                    D_full[:, trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

        motiflet_candidate, motiflet_candidate_dist = get_approximate_k_motiflet(
            data_raw, m, test_k, D_full, upper_bound=upper_bound)

        k_motiflet_distances[test_k] = motiflet_candidate_dist
        k_motiflet_candidates[test_k] = motiflet_candidate
        upper_bound = motiflet_candidate_dist

    # smoothen the line to make it monotonically increasing
    k_motiflet_distances[0:2] = k_motiflet_distances[2]
    for i in range(len(k_motiflet_distances), 2):
        k_motiflet_distances[i - 1] = min(k_motiflet_distances[i],
                                          k_motiflet_distances[i - 1])

    elbow_points = find_elbow_points(k_motiflet_distances)
    return k_motiflet_distances, k_motiflet_candidates, elbow_points, m


@njit
def candidate_dist(D_full, pool, upperbound, m):
    motiflet_candidate_dist = 0
    for i in pool:
        for j in pool:
            if ((i != j and np.abs(i - j) < m / 2)
                    or (i != j and D_full[i, j] > upperbound)):
                return np.inf

    for i in pool:
        for j in pool:
            motiflet_candidate_dist = max(motiflet_candidate_dist, D_full[i, j])

    return motiflet_candidate_dist


# @njit
def find_k_motiflets(ts, D_full, m, k, upperbound=None):
    n = len(ts) - m + 1
    k_halve_m = k * int(m / 2)

    motiflet_dist = upperbound
    if upperbound is None:
        motiflet_candidate, motiflet_dist = get_approximate_k_motiflet(
            ts, m, k, D_full, upper_bound=np.inf)

        motiflet_pos = motiflet_candidate

    # allow subsequence itself
    np.fill_diagonal(D_full, 0)

    for i in range(0, n):
        D_candidates = np.argwhere(D_full[i] <= motiflet_dist).flatten()
        if (len(D_candidates) >= k and
                np.ptp(D_candidates) > k_halve_m):
            # exhaustive search over all subsets
            for permutation in itertools.combinations(D_candidates, k):
                if np.ptp(permutation) > k_halve_m:
                    dist = candidate_dist(D_full, permutation, motiflet_dist, m)
                    if dist < motiflet_dist:
                        motiflet_dist = dist
                        motiflet_pos = np.copy(permutation)

    return motiflet_dist, motiflet_pos