import traceback
import gc
import multiprocessing
import warnings

from motiflets.motiflets import *
from motiflets.plotting import *

warnings.simplefilter("ignore")

# setting for sonic / sone server
num_cores = multiprocessing.cpu_count()
cores = min(64, num_cores - 2)


def test_motiflets_scale_n(
        read_data,
        n_range,  # Time Series length range
        l_range,  # Motif length range
        k_max,
        backends=["default", "pyattimo", "scalable"],
        delta=None,
        subsampling=None,
):
    df = pd.DataFrame(
        columns=['length', 'motif length', 'backend', 'time in s', 'memory in MB', "extent",
                 "motiflet", "elbows"])

    last_time = -1
    for backend in backends:
        results = []

        last_n = 0
        for n in n_range:
            gc.collect()

            print(f"\n\nUsing {backend} size of ts {n} and l_ranges {l_range}")
            print(f"Number of cores {cores}")

            ds_name, ts = read_data()
            if isinstance(ts, pd.DataFrame):
                ts = ts.iloc[:n]
            else:
                ts = ts[:n]
            ts_orig = ts

            if (len(ts_orig) <= last_n
                    # or (backend == "default" and len(ts) > 500_000) \
                    or (last_time > 3600)  # larger than 2 hours
            ):
                break


            if subsampling:
                if isinstance(ts_orig, pd.DataFrame):
                    ts = ts_orig.iloc[::subsampling]
                elif isinstance(ts_orig, pd.Series):
                    ts, _ = compute_paa(ts_orig.to_numpy(), subsampling)
                else:
                    ts, _ = compute_paa(ts_orig, subsampling)


            for l in l_range:
                start = time.time()
                duration = start

                print(f"\n\nRunning: {ds_name}, motif length: {l}, n: {n}")
                l_new = l

                if subsampling:
                    l_new = int(np.round(l / subsampling))
                    print(f"\tApplying Subsampling, Old Size {ts_orig.shape} " +
                          f"New Size {ts.shape}, " +
                          f"New Window Size {l_new}")

                try:
                    mm = Motiflets(
                        ds_name,
                        ts,
                        backend=backend,
                        n_jobs=cores,
                        delta=delta)

                    dists, motiflets, elbow_points = mm.fit_k_elbow(
                        k_max,
                        l_new,
                        plot_elbows=False,
                        plot_motifs_as_grid=False)

                    extent = dists[-1]
                    motiflet = motiflets[-1]

                    if subsampling:
                        motiflet = np.array(motiflet) * subsampling  # scale up again

                    # FIXME ...
                    # if (backend != "pyattimo") and subsampling:    # FIXME add again backend == "pyattimo" or
                    #     # try to refine the positions of the motiflets
                    #     new_motiflet, new_extent = stitch_and_refine(
                    #         ts_orig,
                    #         m=l,
                    #         motiflet=motiflet,
                    #         extent=np.inf if subsampling else extent,
                    #         search_window=min(4*l, 1024),
                    #         # search in a local neighborhood of the motif length
                    #         # upper_bound=extent  # does not work with subsampling
                    #     )
                    #
                    #     if subsampling or new_extent < extent:
                    #         print(f"Searching in local neighborhood. Found a better motif")
                    #         motiflet = new_motiflet
                    #         extent = new_extent

                    backend_name = backend
                    if backend == "pyattimo" and delta is not None:
                        backend_name = f"{backend_name} (delta={delta})"
                    elif subsampling:
                        backend_name = f"{backend_name} (subsampling={subsampling})"

                    duration = time.time() - start
                    memory_usage = mm.memory_usage
                    current = [len(ts_orig), l, backend_name, duration,
                               memory_usage,
                               float(extent),
                               motiflet,
                               elbow_points]

                    results.append(current)
                    df.loc[len(df.index)] = current

                    if backend == "pyattimo":
                        new_filename = f"results/scalability_n_{ds_name}_{k_max}_{backend}_delta_{delta}.csv"
                    elif subsampling:
                        new_filename = f"results/scalability_n_{ds_name}_{k_max}_{backend}_subs_{subsampling}.csv"
                    else:
                        new_filename = f"results/scalability_n_{ds_name}_{backend}_{k_max}.csv"

                    df.to_csv(new_filename, index=False)
                    print(f"\tDiscovered motiflets in {duration:0.2f} seconds")
                    print(f"\t{current}")

                    del mm  # free up memory
                except Exception as e:
                    print(traceback.format_exc())
                except BaseException as e:
                    print(f"Caught a panic: {e}")

                gc.collect()

                last_n = len(ts_orig)
                last_time = duration

        del ts_orig
        del ts

    # print(results)
