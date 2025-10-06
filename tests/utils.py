import traceback
import gc
import warnings

from motiflets.motiflets import *
from motiflets.plotting import *

warnings.simplefilter("ignore")


def force_get(key, kwargs):
    if key in kwargs:
        return kwargs[key]
    else:
        raise ValueError(f"Parameter '{key}' not set")


def test_motiflets_scale_n(
        read_data,
        n_range,  # Time Series length range
        l_range,  # Motif length range
        k_max,
        backends=["default", "pyattimo", "scalable"],
        subsampling=None,
        n_jobs=8,
        **kwargs
):
    for backend in backends:
        df = pd.DataFrame(
            columns=['length', 'motif length', 'backend', 'time in s', 'memory in MB',
                     "extent",
                     "motiflet", "elbows"])

        last_time = -1

        results = []

        last_n = 0
        for n in n_range:
            gc.collect()

            # print(f"\n\tUsing {backend} size of ts {n} and l_ranges {l_range}")
            print(f"\tNumber of cores {n_jobs}")

            ds_name, ts = read_data()
            if isinstance(ts, pd.DataFrame):
                ts = ts.iloc[:n]
            else:
                ts = ts[:n]
            ts_orig = ts

            # larger than 2 hours
            if (len(ts_orig) <= last_n) or (last_time > 3600):
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

                print(f"Running: {ds_name}, motif length: {l}, n: {n}")
                l_new = l

                if subsampling:
                    l_new = int(np.round(l / subsampling))
                    print(f"\tApplying Subsampling {subsampling}, "
                          f"Old Size {ts_orig.shape} " +
                          f"New Size {ts.shape}, " +
                          f"Old Window {l}" +
                          f"New Window {l_new}")

                try:
                    mm = Motiflets(
                        ds_name,
                        ts,
                        backend=backend,
                        n_jobs=n_jobs,
                        **kwargs)

                    dists, motiflets, elbow_points = mm.fit_k_elbow(
                        k_max,
                        l_new,
                        plot_elbows=False,
                        plot_motifs_as_grid=False)

                    extent = dists[-1]
                    motiflet = motiflets[-1]

                    if subsampling:
                        print(f"\tRecomputing Extend using Window Size {l}")
                        motiflet = np.array(motiflet) * subsampling  # scale up again
                        preprocessing = np.array(
                            [mm.distance_preprocessing(ts_orig, l)], dtype=np.float64)

                        extent = get_pairwise_extent_raw(
                            ts_orig.reshape(1, -1), motiflet, l,
                            distance_single=mm.distance_single,
                            preprocessing=preprocessing)

                    backend_name = backend
                    new_filename = f"results/scalability_n_{ds_name}_{k_max}_{backend}"

                    if backend == "pyattimo":
                        pyattimo_delta = force_get("pyattimo_delta", kwargs)
                        backend_name = f"{backend} (delta={pyattimo_delta})"

                        new_filename = (new_filename +
                                        f"_delta_{pyattimo_delta}")

                    elif backend == "pynndescent":
                        pynndescent_n_neighbors = force_get("pynndescent_n_neighbors", kwargs)
                        pynndescent_leaf_size = force_get("pynndescent_leaf_size", kwargs)
                        pynndescent_pruning_degree_multiplier = force_get("pynndescent_pruning_degree_multiplier", kwargs)
                        pynndescent_diversify_prob = force_get("pynndescent_diversify_prob", kwargs)
                        pynndescent_n_search_trees = force_get("pynndescent_n_search_trees", kwargs)
                        pynndescent_search_epsilon = force_get("pynndescent_search_epsilon", kwargs)

                        backend_name = (
                            f"{backend} "
                            f"(n_neighbors={pynndescent_n_neighbors};"
                            f"leaf_size={pynndescent_leaf_size};"
                            f"pruning_degree_multiplier={pynndescent_pruning_degree_multiplier};"
                            f"diversify_prob={pynndescent_diversify_prob};"
                            f"n_search_trees={pynndescent_n_search_trees};"
                            f"search_epsilon={pynndescent_search_epsilon})")

                        new_filename = (
                            new_filename +
                            f"_nn={pynndescent_n_neighbors}"
                            f"_ls={pynndescent_leaf_size}"
                            f"_pdm={pynndescent_pruning_degree_multiplier}"
                            f"_dp={pynndescent_diversify_prob}"
                            f"_nst={pynndescent_n_search_trees}"
                            f"_se={pynndescent_search_epsilon}")

                    elif backend == "faiss":
                        faiss_index = force_get("faiss_index", kwargs)

                        if faiss_index == "HNSW":
                            faiss_efConstruction = force_get("faiss_efConstruction",
                                                             kwargs)
                            faiss_efSearch = force_get("faiss_efSearch", kwargs)
                            faiss_M = force_get("faiss_M", kwargs)

                            backend_name = (f"{backend} "
                                            f"(index={faiss_index};"
                                            f"efConstruction={faiss_efConstruction};"
                                            f"efSearch={faiss_efSearch};"
                                            f"M={faiss_M})")

                            new_filename = (new_filename +
                                            f"_backend_{backend}"
                                            f"_index_{faiss_index}"
                                            f"_efConstruction_{faiss_efConstruction}"
                                            f"_efSearch_{faiss_efSearch}"
                                            f"_M_{faiss_M}")

                        elif faiss_index in ["IVF", "IVFPQ"]:
                            faiss_nprobe = force_get("faiss_nprobe", kwargs)

                            backend_name = (f"{backend} "
                                            f"(index={faiss_index};"
                                            f"faiss_nprobe={faiss_nprobe})")

                            new_filename = (new_filename +
                                            f"_backend_{backend}"
                                            f"_index_{faiss_index}"
                                            f"_faiss_nprobe_{faiss_nprobe}")


                    elif subsampling:
                        backend_name = f"{backend_name} (subsampling={subsampling})"

                        new_filename = (new_filename +
                                        f"_subs_{subsampling}")

                    duration = time.time() - start
                    memory_usage = mm.memory_usage
                    current = [ts_orig.shape[-1],
                               l,
                               backend_name,
                               duration,
                               memory_usage,
                               float(extent),
                               motiflet,
                               elbow_points]

                    results.append(current)
                    df.loc[len(df.index)] = current

                    new_filename = new_filename + ".csv"
                    df.to_csv(new_filename, index=False)

                    print(f"\tDiscovered motiflets in {duration:0.2f} seconds")
                    print("\t'length', 'motif length', 'backend', 'time in s', "
                          "'memory in MB', 'extent', 'motiflet', 'elbows'")
                    # print(f"\t{current}")
                    print("\t" + str(current[0]), *current[1:], sep=', ')

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
