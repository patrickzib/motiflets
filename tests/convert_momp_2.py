import ast
import pandas as pd

import utils as ut
from motiflets.distances import *
from motiflets.motiflets import _sliding_dot_product, _argknn, get_pairwise_extent_raw

from numba import njit, prange


k = 3
momp_path = f"results/momp_converted2/"

datasets = [
    # "EOG_one_hour_50_Hz",
    # "EOG_one_hour_400_Hz",
    # "CinC_Challenge",
    # "swtAttack7",
    # "MGHSleepElectromyography",
    # "recorddata",
    # "Challenge2009Respiration500HZ",
    # "Lab_K_060314",
    # "Bird12-Week3_2018_1_10",
    # "BlackLeggedKittiwake",
    # "water",
    # "FingerFlexionECoG",
    # "SpainishEnergyDataset",
    # "SpainishEnergyDataset5sec",
    # "lorenzAttractorsLONG",
    "stator_winding",
    # "solarwind",
    # "WindTurbine",
    # "SynchrophasorEventsLarge",
    # "Lab_FD_061014",
    # "house",
    # "HAR_Ambient_Sensor_Data",
    # "Challenge2009TestSetA_101a",
    # "swtAttack38",
]

def test_plot():
    df_512 = pd.read_csv("results/momp/momp_512.csv").set_index("dataset")
    df_1024 = pd.read_csv("results/momp/momp_1024.csv").set_index("dataset")
    df_2048 = pd.read_csv("results/momp/momp_2048.csv").set_index("dataset")
    df_4096 = pd.read_csv("results/momp/momp_4096.csv").set_index("dataset")

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

        for motif_length in [512, 1024, 2048, 4096]:

            if (motif_length == 512) :
                try:
                    motiflets = np.fromstring(df_512.loc[ds_name, "motiflet"][1:-1], sep=' ', dtype=np.int32)
                    time = df_512.loc[ds_name, "time in s"]
                except Exception as e:
                    print(f"Skipping {ds_name} for motif length {motif_length} due to error: {e}")
                    continue
            elif motif_length == 1024:
                try:
                    motiflets = np.fromstring(df_1024.loc[ds_name, "motiflet"][1:-1], sep=' ', dtype=np.int32)
                    time = df_1024.loc[ds_name, "time in s"]
                except Exception as e:
                    print(f"Skipping {ds_name} for motif length {motif_length} due to error: {e}")
                    continue
            elif motif_length == 2048:
                try:
                    motiflets = np.fromstring(df_2048.loc[ds_name, "motiflet"][1:-1], sep=' ', dtype=np.int32)
                    time = df_2048.loc[ds_name, "time in s"]
                except Exception as e:
                    print(f"Skipping {ds_name} for motif length {motif_length} due to error: {e}")
                    continue
            elif motif_length == 4096:
                try:
                    motiflets = np.fromstring(df_4096.loc[ds_name, "motiflet"][1:-1], sep=' ', dtype=np.int32)
                    time = df_4096.loc[ds_name, "time in s"]
                except Exception as e:
                    print(f"Skipping {ds_name} for motif length {motif_length} due to error: {e}")
                    continue

            best_motiflet, min_extent = compute_knn(
                ts.copy(),
                motiflets,
                motif_length,
                k - 1)

            current = [
                len(ts),
                motif_length,
                "MOMP",
                time,
                -1,
                best_motiflet,
                min_extent
            ]
            df.loc[len(df.index)] = current

        new_filename = "results/momp_pair/MOMP_" + ds_name + ".csv"
        df.to_csv(new_filename, index=False)


@njit(cache=True)
def compute_knn(
        ts,
        motiflets,
        m,
        k,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        slack=0.5
):
    halve_m = np.int32(m * slack)
    n = ts.shape[-1] - m + 1

    preprocessing = []
    preprocessing.append(distance_preprocessing(ts, m))

    knns = np.zeros((len(motiflets), k), dtype=np.int32)
    extents = np.zeros(len(motiflets), dtype=np.float64)

    a = ts.reshape(1, -1)
    for i in prange(len(motiflets)):
        start = motiflets[i]
        dot_rolled = _sliding_dot_product(
            ts[start:start + m],
            ts,
        )
        dist = distance(dot_rolled, n, m, preprocessing[0], start, halve_m)
        knn = _argknn(dist, k, m, slack=slack)
        knns[i] = knn
        # print(knn)

        extents[i] = get_pairwise_extent_raw(
            a, knns[i], m, distance_single, preprocessing)

    min_pos = np.argmin(extents)
    best_motiflet = knns[min_pos]
    min_extent = extents[min_pos]

    #print(preprocessing[0][1][motiflets[0]])
    #print(preprocessing[0][1][motiflets[1]])

    if len(motiflets) == k:
        extend = get_pairwise_extent_raw(a, motiflets, m, distance_single, preprocessing)
        if min_extent > extend:
            min_extent = extend
            best_motiflet = motiflets

    # print(f"{ds_name} motiflet extent: {min_extent})


    return best_motiflet, min_extent
