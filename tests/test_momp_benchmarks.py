import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/motifs'

import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import traceback
import multiprocessing
import numpy as np
import pandas as pd
import scipy.io as sio
import utils as ut

run_local = True
path = "../datasets/momp/"

filenames = {
    # key, filename, momp motif length, momp motif meaning, dataset length
    "EOG_one_hour_50_Hz": ["EOG_one_hour_50_Hz", 2048, "?", 180000],
    "Challenge2009TestSetA_101a": ["Respiration", 4096, "?", 440001],
    "swtAttack7": ["Swat Attack 7", 4 * 16 * 256, "?", 449919],
    "swtAttack38": ["Swat Attack 38", 16 * 256, "?", 449919],
    "BlackLeggedKittiwake": ["Flying Bird: Black‐legged Kittiwake", 8192, "?", 1288330],
    "stator_winding": ["Electric Motor Temperature", 32768, "?", 1330816],
    "EOG_one_hour_400_Hz": ["EOG_one_hour_400_Hz", 8192, "?", 1439997],
    "Challenge2009Respiration500HZ": ["Challenge 2009 Respiration", 16384, "?", 1799997],
    "HAR_Ambient_Sensor_Data": ["Human Activity Recognition", 4096, "?", 1875227],
    "water": ["Water Demand", 8192, "", 2100777],
    "SpainishEnergyDataset": ["SpainishEnergyDataset", np.nan, "?", 2102701],
    "house": ["Household Electrical Demand", 32768, "?", 5153051],
    "WindTurbine": ["Wind Turbine R24VMON Rotating system", 32768, "Precursor Dropout", 5231008],
    "MGHSleepElectromyography": ["MGH Sleep Electromyography 200 Hz", 32768, "?", 5983000],
    "CinC_Challenge": ["Electroencephalography C3-M2 Part 2", 8192, "Calibration", 6375000],
    "Lab_K_060314": ["ACP on Kryder Citrus", 65536, "", 7583000],
    "Lab_FD_061014": ["Insect EPG - Flaming Dragon", 32768, "?", 7583000],
    "solarwind": ["Solar Wind", 32768, "?", 8066432],
    "Bird12-Week3_2018_1_10": ["22.5 hours of Chicken data at 100 Hz", 16384, "?", 8595817],
    "FingerFlexionECoG": ["Finger Flexion ECoG electrocorticography", 16384, "?", 23999997],
    "SpainishEnergyDataset5sec": ["Spainish Energy Dataset 5 sec", 524288, "?", 25232401],
    "lorenzAttractorsLONG": ["Lorenz Attractors", 524288, "?", 30721281],
    "recorddata": ["EOG Example", 2048, "?", 59430000],
    "SynchrophasorEventsLarge": ["Synchrophasor Events Large", 256 * 2 * 128, "?", 62208000],
}


def read_mat(filename):
    print(f"\tReading {filename} from {path + filename + '.mat'}")
    data = sio.loadmat(path + filename + '.mat')

    # extract data array
    key = filename
    try:
        data = data[key]
    except KeyError:
        # try to find the first key that is not a meta key
        for k in data.keys():
            if ((not k.startswith("__"))
                    and (data[k].dtype in [np.float32, np.float64])):
                key = k
                data = data[k]
                print("\tFound key:", key, "with type", data.dtype)
                break

    # flatten output
    data = pd.DataFrame(data).to_numpy().flatten()
    # data = scipy.stats.zscore(data)

    mb = (data.size * data.itemsize) / (1024 ** 2)

    print(f"\tLength: {len(data)} {mb:0.2f} MB")
    # print(f"\tType: {data.dtype}")
    # print(f"\tContains NaN or Inf? {np.isnan(data).any()} {np.isinf(data).any()}")
    # print(f"\tStats Mean {np.mean(data):0.3f}, Std {np.std(data):0.3f} " +
    #      f"Min {np.min(data):0.3f} Max {np.max(data):0.3f}")

    # remove NaNs
    data = data[~np.isnan(data)]

    # remove Infs
    data = data[np.isfinite(data)]

    # np.savetxt(path + "/csv/" + filename + ".csv", data[:100000], delimiter=",")
    return data.astype(np.float64)


def test_motiflets_scale_n(
        ds_name, data,
        n,
        l_range,
        k_max,
        backends=None,
        subsampling=None,
        n_jobs=8,
        **kwargs
):
    def pass_data():
        return ds_name, data

    ut.test_motiflets_scale_n(
        read_data=pass_data,
        n_range=[n],
        l_range=l_range,
        k_max=k_max,
        backends=backends,
        subsampling=subsampling,
        n_jobs=n_jobs,
        **kwargs
    )


def run_safe(ds_name, series, l_range, k_max, backends,
             subsampling=None, n_jobs=8, **kwargs):
    try:
        if run_local:
            print("\nWarning. Running locally.\n")
            n = 10_000
        else:
            n = len(series)

        test_motiflets_scale_n(
            ds_name, series, n,
            l_range=l_range,
            k_max=k_max,
            backends=backends,
            subsampling=subsampling,
            n_jobs=n_jobs,
            **kwargs)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


# 512 to 8192
if run_local:
    l_range = [512]
else:
    l_range = list([2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13])

# ks = [5, 10, 20]
deltas = [0.1]

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss_hnsw/config.yml
faiss_efConstruction = [500]
faiss_efSearch = [400]  # 600, 800
faiss_M = [64]   # 32, 64, 96

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss/config.yml
# faiss_nlist = [32]  # using sqrt(n) by default
faiss_nprobe = [10]  # [10, 50, 100, 200]

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss/config.yml
# faiss_nlist = [32]  # using sqrt(n) by default
faiss_nprobe = [10]  # [10, 50, 100, 200]

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/pynndescent/config.yml
pynndescent_n_neighbors = [60]
pynndescent_leaf_size = [48]
pynndescent_pruning_degree_multiplier = [2.0]
pynndescent_diversify_prob = [0.0]
pynndescent_n_search_trees = [1]
pynndescent_search_epsilon = [0.2]  # [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36]


k_max = 10

# setting for sonic / sone server
num_cores = multiprocessing.cpu_count()
cores = min(60, num_cores - 2)


def main():
    lengths = np.array([properties[-1] for properties in list(filenames.values())])
    sorted_idx = np.argsort(lengths)
    for filename in np.array(list(filenames.keys()))[sorted_idx]:
        ds_name, length, _, _ = filenames[filename]
        print(f"Running: {filename, ds_name}")
        data = read_mat(filename)

        # # pyattimo
        # backends = ["pyattimo", "scalable"]
        # for delta in deltas:
        #    run_safe(
        #        filename, data, l_range, k_max, backends,
        #        pyattimo_delta=delta, n_jobs=cores
        #    )

        # Running FAISS
        backends = ["faiss"]

        # faiss_index = "HNSW"
        # for M in faiss_M:
        #     for efConstruction in faiss_efConstruction:
        #         for efSearch in faiss_efSearch:
        #             print(f"\n\tRunning faiss {faiss_index} {M} {efConstruction} {efSearch}.", flush=True)
        #             run_safe(
        #                 filename,
        #                 data,
        #                 l_range,
        #                 k_max,
        #                 backends,
        #                 faiss_index=faiss_index,
        #                 faiss_M=M,
        #                 faiss_efConstruction=efConstruction,
        #                 faiss_efSearch=efSearch,
        #                 n_jobs=cores
        #                 )

        # faiss_index = "IVF"
        # for nprobe in faiss_nprobe:
        #     print(f"\n\tRunning faiss {faiss_index} {nprobe}")
        #     run_safe(
        #         filename,
        #         data,
        #         l_range,
        #         k_max,
        #         backends,
        #         faiss_index=faiss_index,
        #         faiss_nprobe=nprobe,
        #         n_jobs=cores
        #      )

        faiss_index = "IVFPQ"
        for nprobe in faiss_nprobe:
            print(f"\n\tRunning faiss {faiss_index} {nprobe}")
            run_safe(
                filename,
                data,
                l_range,
                k_max,
                backends,
                faiss_index=faiss_index,
                faiss_nprobe=nprobe,
                n_jobs=cores
            )


        # # Running pynndescent
        # backends = ["pynndescent"]
        # for n_neighbors in pynndescent_n_neighbors:
        #     for leaf_size in pynndescent_leaf_size:
        #         for pruning_degree_multiplier in pynndescent_pruning_degree_multiplier:
        #             for diversify_prob in pynndescent_diversify_prob:
        #                 for n_search_trees in pynndescent_n_search_trees:
        #                     for search_epsilon in pynndescent_search_epsilon:
        #                         print(f"\n\tRunning pynndescent")
        #                         run_safe(
        #                             filename,
        #                             data,
        #                             l_range,
        #                             k_max,
        #                             backends,
        #                             pynndescent_n_neighbors=n_neighbors,
        #                             pynndescent_leaf_size=leaf_size,
        #                             pynndescent_pruning_degree_multiplier=pruning_degree_multiplier,
        #                             pynndescent_diversify_prob=diversify_prob,
        #                             pynndescent_n_search_trees=n_search_trees,
        #                             pynndescent_search_epsilon=search_epsilon,
        #                             n_jobs=cores
        #                         )



        # scalable
        # backends = ["scalable"]
        # run_safe(
        #   filename, data, l_range, k_max, backends, subsampling=10, n_jobs=cores
        # )

        # # subsampling
        # backends = ["scalable"]
        # for subsampling in [8, 16]:
        #    run_safe(
        #       filename, data,
        #       l_range=l_range, k_max=k_max,
        #       backends=backends,
        #       subsampling=subsampling,
        #       n_jobs=cores
        #    )


if __name__ == "__main__":
    main()
