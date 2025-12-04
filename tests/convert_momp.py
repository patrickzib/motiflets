import pandas as pd

import utils as ut
from motiflets.distances import *
from motiflets.motiflets import _sliding_dot_product, _argknn, get_pairwise_extent_raw

from numba import njit, prange

interesting = {
    "HAR": "HAR_Ambient_Sensor_Data",
    "Household": "house",
    "Solarwind": "solarwind",
    "SpanishED": "SpainishEnergyDataset5sec",
    "Water Demand": "water",
    "Finger ECOG": "FingerFlexionECoG"
}

# MOMP
MOMP_results = {
    # key: [[first_index, second_index], motif_length, time in s]
    "EOG_one_hour_50_Hz": [[62478, 83143], 1024, 23.49],
    "EOG_one_hour_400_Hz": [[236605, 1321975], 8192, 121.17],
    "CinC_Challenge": [[4823203, 4828203], 8192, 999.42],
    "swtAttack7": [[133860, 144969], 16384, 142.10],
    "MGHSleepElectromyography": [[1964375, 5889161], 32768, 3090.66],
    "recorddata": [[3540784, 3998440], 2048, 17370.39],
    "Challenge2009Respiration500HZ": [[62963, 1296128], 16384, 562.95],
    "Lab_K_060314": [[6817003, 6979308], 65536, 55744.02],
    "Bird12-Week3_2018_1_10": [[80420, 1374748], 16384, 9324.53],
    "BlackLeggedKittiwake": [[45696, 742777], 8192, 431.02],
    "water": [[973390, 1255811], 8192, 475.08],
    "FingerFlexionECoG": [[14395771, 19195771], 16384, 10859.10],
    "SpainishEnergyDataset5sec": [[12884629, 13368457], 524288, 44034.65],
    "lorenzAttractorsLONG": [[1, 15361793], 524288, 48210.02],
    "stator_winding": [[336089, 392549], 32768, 119.93],
    "solarwind": [[2221814, 5071908], 32768, 1049.56],
    "WindTurbine": [[929205, 1226751], 32768, 5906.36],
    "SynchrophasorEventsLarge": [[30610002, 61714007], 65536, 17159.20],
    "Lab_FD_061014": [[6684313, 7342637], 32768, 881.47],
    "house": [[617128, 928816], 32768, 750.50],
    "HAR_Ambient_Sensor_Data": [[425966, 1615881], 4096, 351.91],
    "Challenge2009TestSetA_101a": [[353601, 424237], 4096, 300.95],
    "swtAttack38": [[311254, 319874], 4096, 307.88]
}

k = 10
momp_path = f"results/pyattimo_momp/k20/"


def test_plot():
    for ds_name, _ in MOMP_results.items():
        #if f"{ds_name}.json" in os.listdir(momp_path):
        #    json_path = f"{momp_path}{ds_name}.json"
        #    df = pd.read_json(json_path)

        df = pd.DataFrame(columns=[
            "length",
            "motif length",
            "backend",
            "time in s",
            "memory in MB",
            "motiflet",
            "extent"])

        ts = ut.read_mat(ds_name)

        momp = MOMP_results[ds_name]
        motiflets = np.array(momp[0], dtype=np.int32)
        # motif_length = np.int32(momp[1])
        time = momp[2]

        for motif_length in [512, 1024, 2048, 4096]:
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

        new_filename = "csv/MOMP_" + ds_name + ".csv"
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
    # print(f"{ds_name} motiflet extent: {min_extent}")

    return best_motiflet, min_extent
