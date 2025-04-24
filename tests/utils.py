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
        n_range,  # Time Series length range
        l_range,    # Motif length range
        k_max,
        backends=["default", "pyattimo", "scalable"],
        delta = None,
        subsampling=None,
    ):
    df = pd.DataFrame(columns=['length', 'backend', 'time in s', 'memory in MB', "extent", "motiflet"])

    last_time = -1
    results = []
    for backend in backends:
        last_n = 0
        for n in n_range:
            gc.collect()

            start = time.time()
            print(f"\tUsing {backend} size of ts {n}")

            ds_name, ts = read_data()
            if isinstance(ts,pd.DataFrame):
                ts = ts.iloc[:n]
            else:
                ts = ts[:n]
            ts_orig = ts

            if (len(ts_orig) <= last_n
                    or (backend == "default" and len(ts) > 500_000) \
                    or (last_time > 3600)     # larger than 2 hours
            ):
                break

            # print("Size of TS: ", ts.shape)

            for l in l_range:
                l_new = l
                if subsampling:
                    if isinstance(ts_orig, pd.DataFrame):
                        ts = ts_orig.iloc[::subsampling]
                    elif isinstance(ts_orig, pd.Series):
                        ts, _ = compute_paa(ts_orig.to_numpy(), subsampling)
                    else:
                        ts, _ = compute_paa(ts_orig, subsampling)

                    l_new = int(np.round(l / subsampling))
                    print(f"\tApplying Subsampling, Old Size {ts_orig.shape} " +
                          f"New Size {ts.shape}, " +
                          f"New Window Size {l_new}")

                print(f"\tNumber of cores {cores}")
                mm = Motiflets(
                    ds_name,
                    ts,
                    backend=backend,
                    n_jobs=cores,
                    delta=delta)

                dists, motiflets, _ = mm.fit_k_elbow(
                    k_max,
                    l_new,
                    plot_elbows=False,
                    plot_motifs_as_grid=False)

                extent = dists[-1]
                motiflet = motiflets[-1]

                if subsampling:
                    motiflet = np.array(motiflet) * subsampling   # scale up again

                if backend == "pyattimo" or subsampling:
                    # try to refine the positions of the motiflets
                    new_motiflet, new_extent = stitch_and_refine(
                        ts_orig,
                        m=l,
                        motiflet=motiflet,
                        extent=np.inf if subsampling else extent,
                        search_window=l * 4,  # search in a local neighborhood of 4 times the motif length
                        # upper_bound=extent  # does not work with subsampling
                    )

                    if subsampling or new_extent < extent:
                        print(f"Searching in local neighborhood. Found a better motif")
                        motiflet = new_motiflet
                        extent = new_extent

                backend_name = backend
                if backend == "pyattimo" and delta is not None:
                    backend_name = f"{backend_name} (delta={delta}, v=0.6.4)"
                elif backend == "pyattimo":
                    backend_name = f"{backend_name} (v=0.6.4)"
                elif subsampling:
                    backend_name = f"{backend_name} (subsampling={subsampling})"

                duration = time.time() - start
                memory_usage = mm.memory_usage
                current = [len(ts_orig), backend_name, duration, memory_usage, extent, motiflet]

                results.append(current)
                df.loc[len(df.index)] = current

                if backend == "pyattimo":
                    if delta:
                        new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}_pyattimo_delta_{delta}.csv"
                    else:
                        new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}_pyattimo_delta_{delta}.csv"
                elif subsampling:
                    new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}_subs_{subsampling}.csv"
                else:
                    new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}.csv"

                df.to_csv(new_filename, index=False)
                print("\tDiscovered motiflets in", duration, "seconds")
                print("\t", current[-1])

                del mm  # free up memory
                gc.collect()

                last_n = len(ts_orig)
                last_time = duration

    # print(results)