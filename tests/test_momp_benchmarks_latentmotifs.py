import os
import time

os.environ['NUMBA_CACHE_DIR'] = '/tmp/motifs'

import psutil
import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import traceback
import multiprocessing
import pandas as pd
import scipy.io as sio
from competitors.latentmotifs import *
from motiflets.distances import *
from motiflets.motiflets import get_pairwise_extent_raw

run_local = True
path = "/vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp/"
if os.path.exists(path) and os.path.isdir(path):
    run_local = False
else:
    path = "../datasets/momp/"

print(f"Using directory: {path} {run_local}")

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



def run_safe(
        ds_name,
        series,
        k_max,
        l_range,
        n_jobs=8
):
    new_filename = f"results/scalability_n_{ds_name}_{k_max}_latentmotifs"
    new_filename = new_filename + ".csv"

    df_results = pd.DataFrame(
        columns=[
            'length', 'motif length', 'backend',
            'time in s', 'memory in MB',
            'extent', 'motiflet', 'elbows'
        ])

    results = []
    try:
        if run_local:
            print("\nWarning. Running locally.\n")
            n = 10_000
        else:
            n = len(series)

        results_name = f"results/pyattimo_0.7.0_20GB/scalability_n_{ds_name}_{k_max}_pyattimo_delta_0.1.csv"
        df = pd.read_csv(results_name)[["motif length", "extent"]]
        df.set_index("motif length", inplace=True)

        for length in l_range:
            ts = series[:n]
            print(f"Running LatentMotif for length {length} ")
            extent = (df.loc[length]["extent"])*2
            print(f"\tExtent {extent}")

            pid = os.getpid()
            process = psutil.Process(pid)

            start = time.time()
            lm = LatentMotif(
                n_patterns=1,
                wlen=length,
                radius=extent,
            )
            lm.fit(ts)
            duration = time.time() - start

            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

            print(f"\tDiscovered motiflets in {duration:0.2f} seconds")
            print(f"\tMemory usage: {memory_usage:0.2f} MB")

            motif_set = np.array(lm.prediction_mask_[1])[0]
            print(f"\tPatterns {lm.patterns_.shape[0]}")
            print(f"\tLocations {motif_set.shape[0]}")

            distance_preprocessing, _, distance_single = map_distances("znormed_ed")
            preprocessing = np.array([distance_preprocessing(ts, length)], dtype=np.float64)

            if len(motif_set)>0:
                extent = get_pairwise_extent_raw(
                    ts.reshape(1, -1), motif_set, length,
                    distance_single=distance_single,
                    preprocessing=preprocessing)
            else:
                extent = np.inf

            print(f"\tExtent {extent}")
            del lm

            current = [
                ts.shape[-1],
                length,
                "LatentMotif",
                duration,
                memory_usage,
                float(extent),
                motif_set,
                -1
            ]

            results.append(current)
            df_results.loc[len(df_results.index)] = current
            df_results.to_csv(new_filename, index=False)

    except Exception as e:
        print(f"Caught a panic: {e}")
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


# 512 to 8192
if run_local:
    l_range = [2**9]
else:
    l_range = list([2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13])

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

        run_safe(
            filename, data, k_max, l_range, n_jobs=cores
        )

if __name__ == "__main__":
    main()
