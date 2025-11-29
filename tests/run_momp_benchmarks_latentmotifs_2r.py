import os

os.environ['NUMBA_CACHE_DIR'] = '/tmp/motifs'

import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import time
import psutil
import traceback
import pandas as pd
import numpy as np
import utils as ut

from competitors.latentmotifs import LatentMotif
from motiflets.motiflets import get_pairwise_extent_raw
from motiflets.distances import map_distances

run_local = True
path = "/vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp/"
if os.path.exists(path) and os.path.isdir(path):
    run_local = False


def run_safe(
        ds_name,
        series,
        k_max,
        l_range
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

        path1 = f"results/latentmotifs_2r/scalability_n_{ds_name}_10_latentmotifs.csv"
        # path2 = f"results/latentmotifs_r2/scalability_n_{ds_name}_10_latentmotifs.csv"
        lm_extents = np.full(len(l_range), np.inf, dtype=np.float64)

        if os.path.exists(path1):
            latent = pd.read_csv(path1)
            lm_extents = latent["extent"].to_numpy()

        for i, length in enumerate(l_range):
            if np.isinf(lm_extents[i]):
                ts = series[:n]
                print(f"Running LatentMotif for length {length} ")

                # 2r
                extent = (df.loc[length]["extent"] + 1e-4) * 2
                print(f"\tExtent {extent}")

                pid = os.getpid()
                process = psutil.Process(pid)

                start = time.time()
                lm = LatentMotif(
                    n_patterns=1,
                    wlen=length,
                    radius=extent,
                    n_starts=10,
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
                preprocessing = np.array([distance_preprocessing(ts, length)],
                                         dtype=np.float64)

                if len(motif_set) > 0:
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
                    "LatentMotif 2r",
                    duration,
                    memory_usage,
                    float(extent),
                    motif_set,
                    -1
                ]

                results.append(current)
                df_results.loc[len(df_results.index)] = current
                df_results.to_csv(new_filename, index=False)

            else:
                print(f"Skipping LatentMotif for length {length} ")

    except Exception as e:
        print(f"Caught a panic: {e}")
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


# 512 to 8192
if run_local:
    l_range = [2 ** 9]
else:
    l_range = list([2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12])

k_max = 10


def main():
    lengths = np.array([properties[-1] for properties in list(ut.filenames.values())])
    sorted_idx = np.argsort(lengths)
    for filename in np.array(list(ut.filenames.keys()))[sorted_idx]:
        ds_name, length, _, _ = ut.filenames[filename]
        print(f"Running: {filename, ds_name}")
        data = ut.read_mat(filename)

        run_safe(
            filename, data, k_max, l_range
        )


if __name__ == "__main__":
    main()
