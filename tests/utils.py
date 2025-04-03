from datetime import datetime

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
            ts = ts.iloc[:n]

            if (len(ts) <= last_n
                    or (backend == "default" and len(ts) > 500_000) \
                    or (last_time > 3600)     # larger than 2 hours
            ):
                break

            print("Size of DS: ", ts.shape)

            mm = Motiflets(
                ds_name, ts, backend=backend, n_jobs=cores, delta=delta)

            dists, motiflets, _ = mm.fit_k_elbow(
                k_max, l, plot_elbows=False,
                plot_motifs_as_grid=False)

            duration = time.time() - start
            memory_usage = mm.memory_usage
            extent = dists[-1]
            motiflet = motiflets[-1]


            if backend == "pyattimo":
                # try to refine the positions of the motiflets
                new_motiflet, new_extent = stitch_and_local_motiflet_search(
                    ts,
                    l,
                    motiflet,
                    extent,
                    l * 4,
                    upper_bound=extent
                )

                if new_extent < extent:
                    print(f"Searching in local neighborhood found a better motif")
                    motiflet = new_motiflet
                    extent = new_extent

            current = [len(ts), backend, duration, memory_usage, extent, motiflet]

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