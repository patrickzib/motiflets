from datetime import datetime

import pandas as pd

from motiflets.plotting import *
from motiflets.motiflets import *

import multiprocessing
import gc
import warnings
warnings.simplefilter("ignore")

# setting for sonic / sone server
num_cores = multiprocessing.cpu_count()
cores = min(64, num_cores - 2)

def test_motiflets_scale_n(
        read_data,
        length_range,
        l, k_max,
        backends=["default", "pyattimo", "scalable"],
        delta = None,
        subsampling=None,
        ):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    df = pd.DataFrame(columns=['length', 'backend', 'time in s', 'memory in MB', "extent", "motiflet"])

    last_time = -1
    results = []
    for backend in backends:
        last_n = 0
        for n in length_range:
            start = time.time()
            print(backend, n)

            ds_name, ts = read_data()
            if isinstance(ts,pd.DataFrame):
                ts = ts.iloc[:n]
            else:
                ts = ts[:n]

            if (len(ts) <= last_n
                    or (backend == "default" and len(ts) > 500_000) \
                    or (last_time > 3600)     # larger than 2 hours
            ):
                break

            print("Size of TS: ", ts.shape)

            l_new = l
            if subsampling:
                if isinstance(ts,pd.DataFrame):
                    ts = ts.iloc[::subsampling]
                else:
                    ts = ts[::subsampling]

                l_new = int(l / subsampling)
                print("Applying Subsampling, New Size:", ts.shape)

            mm = Motiflets(
                ds_name, ts, backend=backend, n_jobs=cores, delta=delta)

            dists, motiflets, _ = mm.fit_k_elbow(
                k_max, l_new, plot_elbows=False,
                plot_motifs_as_grid=False)

            duration = time.time() - start
            memory_usage = mm.memory_usage
            extent = dists[-1]
            motiflet = motiflets[-1]

            if subsampling:
                motiflet = motiflet * subsampling   # scale up again

            if backend == "pyattimo" or subsampling:
                # try to refine the positions of the motiflets
                new_motiflet, new_extent = stitch_and_local_motiflet_search(
                    ts,
                    l,
                    motiflet,
                    extent,
                    l * 4,  # search in a local neighborhood of 4 times the motif length
                    # upper_bound=extent  # does not work with subsampling
                )

                print(f"Searching in local neighborhood, found a better motif")
                motiflet = new_motiflet
                extent = new_extent

            backend_name = backend
            if backend == "pyattimo" and delta is not None:
                backend_name = f"{backend_name} (delta={delta})"
            elif subsampling:
                backend_name = f"{backend_name} (subsampling={subsampling})"

            current = [len(ts), backend_name, duration, memory_usage, extent, motiflet]

            results.append(current)
            df.loc[len(df.index)] = current

            if delta:
                new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}_{delta}_{timestamp}.csv"
            else:
                new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}_{timestamp}.csv"

            df.to_csv(new_filename, index=False)
            print("\tDiscovered motiflets in", duration, "seconds")
            print("\t", current)

            gc.collect()

            last_n = len(ts)
            last_time = duration

    print(results)