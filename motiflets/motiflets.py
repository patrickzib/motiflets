import itertools
from ast import literal_eval
from os.path import exists

import numpy as np
import numpy.fft as fft
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import zscore
from tqdm import tqdm

slack = 0.6


def as_series(data, index_range, index_name):
    series = pd.Series(data=data, index=index_range)
    series.index.name = index_name
    return series


def resample(data, sampling_factor=10000):
    factor = 1
    if len(data) > sampling_factor:
        factor = np.int32(len(data) / sampling_factor)
        data = data[::factor]
    return data, factor


def read_ground_truth(dataset):
    file = '../datasets/ground_truth/' + dataset.split(".")[0] + "_gt.csv"
    if exists(file):
        print(file)
        series = pd.read_csv(
            file, index_col=0,
            converters={1: literal_eval, 2: literal_eval, 3: literal_eval})
        return series
    return None


def read_dataset_with_index(dataset, sampling_factor=10000):
    full_path = '../datasets/ground_truth/' + dataset
    data = pd.read_csv(full_path, index_col=0, squeeze=True)
    print("Dataset Original Length n: ", len(data))

    data, factor = resample(data, sampling_factor)
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


def read_dataset(dataset, sampling_factor=10000):
    full_path = '../datasets/' + dataset
    data = pd.read_csv(full_path).T
    data = np.array(data)[0]
    print("Dataset Original Length n: ", len(data))

    data, factor = resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    # gt = read_ground_truth(dataset)    
    return zscore(data)  # , gt


def read_segmenation_ts(file):
    path = "../datasets/tssb/"
    ts = pd.read_csv(path + file, header=None)

    parts = file.split(".")[0].split("_")
    true_cps = np.int32(parts[2:])
    period_size = int(parts[1])

    ds_name = parts[0]

    # _ = plot_time_series_with_change_points(ds_name, ts, true_cps)
    return ts[0], period_size, true_cps, ds_name


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
    halve_m = int(m * slack)

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


@njit(fastmath=True, cache=True)
def get_radius(D_full, motiflet_pos, upperbound=np.inf):
    """ Requires the full matrix!!! """

    motiflet_radius = np.inf

    for ii in range(len(motiflet_pos) - 1):
        i = motiflet_pos[ii]
        current = np.float32(0.0)
        for jj in range(1, len(motiflet_pos)):
            if (i != jj):
                j = motiflet_pos[jj]
                current = max(current, D_full[i, j])
        motiflet_radius = min(current, motiflet_radius)

    return motiflet_radius


@njit(fastmath=True, cache=True)
def get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf):
    """ Requires the full matrix!!! """

    motiflet_extent = np.float32(0.0)
    for ii in range(len(motiflet_pos) - 1):
        i = motiflet_pos[ii]
        for jj in range(ii + 1, len(motiflet_pos)):
            j = motiflet_pos[jj]

            motiflet_extent = max(motiflet_extent, D_full[i, j])
            if motiflet_extent >= upperbound:
                return np.inf

    return motiflet_extent


@njit(fastmath=True, cache=True)
def get_top_k_non_trivial_matches_inner(
        dist, k, candidates, lowest_dist=np.inf):
    # admissible pruning: are there enough offsets within range?    
    if (len(candidates) < k):
        return candidates

    dists = np.copy(dist)
    idx = []  # there may be less than k, thus use a list
    for i in range(len(candidates)):
        pos = candidates[np.argmin(dists[candidates])]
        if dists[pos] < lowest_dist:
            idx.append(pos)
            dists[pos] = np.inf
        else:
            break

    return np.array(idx, dtype=np.int32)


@njit(fastmath=True, cache=True)
def get_top_k_non_trivial_matches(
        dist, k, m, n, lowest_dist=np.inf):
    dist_idx = np.argwhere(dist < lowest_dist).flatten().astype(np.int32)
    # not possible, as wehave to check for overlapps, too
    # if (len(dist_idx) <= k):
    #    return dist_idx

    halve_m = int(m * slack)
    dists = np.copy(dist)
    idx = []  # there may be less than k, thus use a list
    for i in range(k):
        pos = dist_idx[np.argmin(dists[dist_idx])]
        if (not np.isnan(dists[pos])) and (dists[pos] < lowest_dist):
            idx.append(pos)
            dists[max(0, pos - halve_m):min(pos + halve_m, n)] = np.inf
        else:
            break

    return np.array(idx, dtype=np.int32)


# @njit
def get_approximate_k_motiflet(
        ts, m, k, D,
        upper_bound=np.inf, incremental=False, all_candidates=None
):
    n = len(ts) - m + 1
    motiflet_dist = upper_bound
    motiflet_candidate = None

    motiflet_all_candidates = np.zeros(n, dtype=object)

    # allow subsequence itself
    np.fill_diagonal(D, 0)

    # TODO: parallelize??
    for i, order in enumerate(np.arange(n)):
        dist = np.copy(D[order])

        if incremental:
            idx = get_top_k_non_trivial_matches_inner(
                dist, k, all_candidates[order], motiflet_dist)
        else:
            idx = get_top_k_non_trivial_matches(dist, k, m, n, motiflet_dist)

        motiflet_all_candidates[i] = idx

        if len(idx) >= k and dist[idx[-1]] <= motiflet_dist:
            # get_pairwise_extent requires the full matrix 
            motiflet_extent = get_pairwise_extent(D, idx[:k], motiflet_dist)
            if motiflet_dist > motiflet_extent:
                motiflet_dist = motiflet_extent
                motiflet_candidate = idx[:k]

    return motiflet_candidate, motiflet_dist, motiflet_all_candidates


@njit(fastmath=True, cache=True)
def check_unique(elbow_points_1, elbow_points_2, motif_length):
    count = 0
    for a in elbow_points_1:  # smaller motiflet
        for b in elbow_points_2:  # larger motiflet
            if abs(a - b) < (motif_length / 4):
                count = count + 1
                break

        if count >= len(elbow_points_1) / 2:
            return False
    return True


def filter_unique(elbow_points, candidates, motif_length):
    filtered_ebp = []
    for i in range(len(elbow_points)):
        unique = True
        for j in range(i + 1, len(elbow_points)):
            unique = check_unique(
                candidates[elbow_points[i]], candidates[elbow_points[j]], motif_length)
            if not unique:
                break
        if unique:
            filtered_ebp.append(elbow_points[i], )

    # print("Elbows", filtered_ebp)
    return np.array(filtered_ebp)


@njit(fastmath=True, cache=True)
def find_elbow_points(dists, alpha=2):
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
        if peaks[p] > alpha:
            elbow_points.append(p)
            peaks[p - 1:p + 2] = 0
        else:
            break

    return np.sort(np.array(list(set(elbow_points))))


def inner_au_pef(data, k_max, m, upper_bound):
    dists, candidates, elbow_points, _ = search_k_motiflets_elbow(
        k_max,
        data,
        m,
        upper_bound=upper_bound)

    if np.isnan(dists).any() or np.isinf(dists).any():
        return None, None, None

    au_pefs = ((dists - dists.min()) / (dists.max() - dists.min())).sum() / len(dists)
    elbow_points = filter_unique(elbow_points, candidates, m)

    top_motiflet = None
    if len(elbow_points > 0):
        elbows = len(elbow_points)
        top_motiflet = candidates[elbow_points[-1]]
    else:
        # pair motif
        elbows = 1
        top_motiflet = candidates[0]

    return au_pefs, elbows, top_motiflet, dists


def find_au_pef_motif_length(data, k_max, motif_length_range):
    # apply sampling for speedup only
    subsample = 2
    data = data[::subsample]

    index = (data.index / subsample) if isinstance(data, pd.Series) else np.arange(
        len(data))

    # in reverse order
    au_pefs = np.zeros(len(motif_length_range), dtype=object)
    elbows = np.zeros(len(motif_length_range), dtype=object)
    top_motiflets = np.zeros(len(motif_length_range), dtype=object)

    upper_bound = np.inf
    for i, m in enumerate(motif_length_range[::-1]):
        au_pefs[i], elbows[i], top_motiflets[i], dist = inner_au_pef(
            data, k_max, int(m / subsample),
            upper_bound=upper_bound)
        upper_bound = min(dist[-1], upper_bound)

    au_pefs = np.array(au_pefs, dtype=np.float64)[::-1]
    elbows = elbows[::-1]
    top_motiflets = top_motiflets[::-1]

    # if no elbow can be found, ignore this part
    condition = np.argwhere(elbows == 0).flatten()
    au_pefs[condition] = np.inf

    return motif_length_range[np.nanargmin(au_pefs)], au_pefs, elbows, top_motiflets


def search_k_motiflets_elbow(
        k_max,
        data,
        motif_length='auto',
        motif_length_range=None,
        exclusion=None,
        upper_bound=np.inf):
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
            data, k_max, motif_length_range)
    elif isinstance(motif_length, int) or \
            isinstance(motif_length, np.int32) or \
            isinstance(motif_length, np.int64):
        m = motif_length
    else:
        print("Warning: no valid motif_length set - use 'auto' for automatic selection")
        assert False

    k_motiflet_distances = np.zeros(k_max)
    k_motiflet_candidates = np.empty(k_max, dtype=object)

    D_full = compute_distances_full(data_raw, m)

    exclusion_m = int(m * slack)
    motiflet_candidates = []

    for test_k in tqdm(range(k_max - 1, 1, -1), desc='Compute ks'):
        # Top-N retrieval
        if exclusion is not None and exclusion[test_k] is not None:
            for pos in exclusion[test_k].flatten():
                if pos is not None:
                    trivialMatchRange = (max(0, pos - exclusion_m),
                                         min(pos + exclusion_m, len(D_full)))
                    D_full[:, trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

        incremental = (test_k < k_max - 1)
        candidate, candidate_dist, all_candidates = get_approximate_k_motiflet(
            data_raw, m, test_k, D_full,
            upper_bound=upper_bound,
            incremental=incremental,  # we use an incremental computation
            all_candidates=motiflet_candidates
        )

        if len(motiflet_candidates) == 0:
            motiflet_candidates = all_candidates

        k_motiflet_distances[test_k] = candidate_dist
        k_motiflet_candidates[test_k] = candidate
        upper_bound = candidate_dist

    # smoothen the line to make it monotonically increasing
    k_motiflet_distances[0:2] = k_motiflet_distances[2]
    for i in range(len(k_motiflet_distances), 2):
        k_motiflet_distances[i - 1] = min(k_motiflet_distances[i],
                                          k_motiflet_distances[i - 1])

    elbow_points = find_elbow_points(k_motiflet_distances)
    return k_motiflet_distances, k_motiflet_candidates, elbow_points, m


@njit(fastmath=True, cache=True)
def candidate_dist(D_full, pool, upperbound, m):
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


# @njit
def find_k_motiflets(ts, D_full, m, k, upperbound=None):
    n = len(ts) - m + 1

    motiflet_dist = upperbound
    if upperbound is None:
        motiflet_candidate, motiflet_dist, _ = get_approximate_k_motiflet(
            ts, m, k, D_full, upper_bound=np.inf)

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
                        dist = candidate_dist(D_full, permutation, motiflet_dist, m)
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
