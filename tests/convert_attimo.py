import os.path
import pandas as pd

import utils as ut
from motiflets.distances import *
from motiflets.motiflets import _sliding_dot_product, _argknn, get_pairwise_extent_raw

from numba.typed import List
from numba import njit, prange

k = 10
momp_path = f"results/pyattimo_momp/k20/"

datasets = [
    "EOG_one_hour_50_Hz",
    "EOG_one_hour_400_Hz",
    "CinC_Challenge",
    "swtAttack7",
    "MGHSleepElectromyography",
    "recorddata",
    "Challenge2009Respiration500HZ",
    "Lab_K_060314",
    "Bird12-Week3_2018_1_10",
    "BlackLeggedKittiwake",
    "water",
    "FingerFlexionECoG",
    "SpainishEnergyDataset",
    "SpainishEnergyDataset5sec",
    "lorenzAttractorsLONG",
    "stator_winding",
    "solarwind",
    "WindTurbine",
    "SynchrophasorEventsLarge",
    "Lab_FD_061014",
    "house",
    "HAR_Ambient_Sensor_Data",
    "Challenge2009TestSetA_101a",
    "swtAttack38",
]

def parse_array(cell):
    # Strip brackets and parse with space separator
    cleaned = cell.strip('[]')
    array = np.fromstring(cleaned, sep=' ')
    return array.astype(np.int64)  # Convert to int if no decimals expected

def flatten_with_empties(arr):
    non_empty = [sub for sub in arr if len(sub) > 0]
    return np.concatenate(non_empty) if non_empty else np.array([])

def test_plot():
    for ds_name in datasets:

        df = pd.DataFrame(columns=[
            "length",
            "motif length",
            "backend",
            "time in s",
            "memory in MB",
            "motiflet",
            "extent"])

        ts = ut.read_mat(ds_name)

        path1 = f"results/attimo/scalability_n_{ds_name}_3_pyattimo_delta_0.1.csv"
        df_attimo = pd.read_csv(path1)
        df_attimo = df_attimo.set_index("motif length")

        for i, motif_length in enumerate([512, 1024, 2048, 4096]):
            print(f"Processing dataset {ds_name} with motif length {motif_length}")

            if motif_length in df_attimo.index:
                motiflets = parse_array(df_attimo.loc[motif_length, "motiflet"])
                best_motiflet, min_extent = compute_knn(
                    ts.copy(),
                    motiflets.copy(),
                    motif_length,
                    k - 1
                )
            else:
                best_motiflet = []
                min_extent = np.inf

            if motif_length in df_attimo.index:
                time = df_attimo["time in s"].values[i]
                memory = df_attimo["memory in MB"].values[i]
            else:
                last_time = df_attimo["time in s"].values[-1]
                last_memory = df_attimo["memory in MB"].values[-1]
                last_length = df_attimo["motif length"].values[-1]
                factor = motif_length / last_length
                time = last_time * factor * 2
                memory = last_memory * factor

                print(f"Extrapolating time and memory for motif length {motif_length} "
                      f"based on last known length {last_length}")
                print(f"Factor: {factor}")
                print(f"\tEstimated time: {time}, Estimated memory: {memory}")

            current = [
                len(ts),
                motif_length,
                "attimo",
                time,
                memory,
                best_motiflet,
                min_extent
            ]
            df.loc[len(df.index)] = current

        new_filename = f"results/attimo_converted/scalability_n_{ds_name}_{k}_attimo.csv"
        df.to_csv(new_filename, index=False)


@njit(cache=True)
def compute_knn(
        ts,
        motiflets,
        m,
        k,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
):
    halve_m = np.int32(m * slack)
    n = ts.shape[-1] - m + 1

    preprocessing = distance_preprocessing(ts, m)

    knns = np.zeros((len(motiflets), k), dtype=np.int32)
    extents = np.zeros(len(motiflets), dtype=np.float64)

    for i in prange(len(motiflets)):
        start = motiflets[i]
        dot_rolled = _sliding_dot_product(
            ts[start:start + m],
            ts,
        )
        if start < len(ts) - m + 1:
            dist = distance(dot_rolled, n, m, preprocessing, start, halve_m)
            knn = _argknn(dist, k, m, slack=slack)
            knns[i] = knn
            extents[i] = get_pairwise_extent_raw_1d(
                ts, knns[i], m, distance_single, preprocessing)
        else:
            extents[i] = np.inf

    min_pos = np.argmin(extents)
    best_motiflet = knns[min_pos]
    min_extent = extents[min_pos]
    # print(f"{ds_name} motiflet extent: {min_extent}")

    return best_motiflet, min_extent


@njit(cache=True)
def get_pairwise_extent_raw_1d(
        series, motifset_pos, motif_length,
        distance_single, preprocessing):
    """Computes the extent of the motifset via pairwise comparisons.

    Parameters
    ----------
    series : array-like
        The time series
    motifset_pos : array-like
        The motif set start-offsets
    motif_length : int
        The motif length
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

    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        a = series[i:i + motif_length]

        for jj in np.arange(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]
            b = series[j:j + motif_length]
            dist = distance_single(a, b, i, j, preprocessing)
            motifset_extent = max(motifset_extent, dist)

    return motifset_extent