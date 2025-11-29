import gc
import traceback
import warnings
import multiprocessing

import scipy.io as sio

from motiflets.motiflets import *
from motiflets.plotting import *

warnings.simplefilter("ignore")

run_local = True
path = "/vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp/"
if os.path.exists(path) and os.path.isdir(path):
    run_local = False
else:
    path = "../datasets/momp/"

print(f"Using directory: {path} {run_local}")

filenames = {
    # key, filename, momp motif length, momp motif meaning, dataset length
    "EOG_one_hour_50_Hz": ["EOG_one_hour_50_Hz", 1024, "?", 180000],
    "Challenge2009TestSetA_101a": ["Respiration", 4096, "?", 440001],
    "swtAttack7": ["Swat Attack 7", 16384, "?", 449919],
    "swtAttack38": ["Swat Attack 38", 4096, "?", 449919],
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
    "SynchrophasorEventsLarge": ["Synchrophasor Events Large", 65536, "?",62208000],
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
        ds_name, series, l_range, k_max,
        backend, subsampling=None, n_jobs=-1, **kwargs):
    try:
        if run_local:
            print("\nWarning. Running locally.\n")
            n = 10_000
        else:
            n = len(series)

        def pass_data():
            return ds_name, series

        test_motiflets_scale_n(
            read_data=pass_data,
            n_range=[n],
            l_range=l_range,
            k_max=k_max,
            backend=backend,
            subsampling=subsampling,
            n_jobs=n_jobs,
            **kwargs
        )

    except Exception as e:
        print(f"Caught a panic: {e}")
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


def force_get(key, kwargs):
    if key in kwargs:
        return kwargs[key]
    else:
        raise ValueError(f"Parameter '{key}' not set")


def find_dominant_window_sizes(X, offset=0.05):
    """Determine the Window-Size using dominant FFT-frequencies.

    Parameters
    ----------
    X : array-like, shape=[n]
        a single univariate time series of length n
    offset : float
        Exclusion Radius

    Returns
    -------
    trivial_match: bool
        If the candidate change point is a trivial match
    """
    fourier = np.absolute(np.fft.fft(X))
    freqs = np.fft.fftfreq(X.shape[0], 1)

    coefs = []
    window_sizes = []

    for coef, freq in zip(fourier, freqs):
        if coef and freq > 0:
            coefs.append(coef)
            window_sizes.append(1 / freq)

    coefs = np.array(coefs)
    window_sizes = np.asarray(window_sizes, dtype=np.int64)

    idx = np.argsort(coefs)[::-1]
    return next(
        (
            int(window_size / 2)
            for window_size in window_sizes[idx]
            if window_size in range(20, int(X.shape[0] * offset))
        ),
        window_sizes[idx[0]],
    )


def test_motiflets_scale_n(
        read_data,
        n_range,  # Time Series length range
        l_range,  # Motif length range
        k_max,
        backend="pyattimo",
        subsampling=None,
        n_jobs=-1,
        **kwargs
):
    if n_jobs == -1:
        num_cores = multiprocessing.cpu_count()

        # tuned for sonic / sone server
        n_jobs = min(60, num_cores - 2)

    df = pd.DataFrame(
        columns=['length', 'motif length', 'backend', 'time in s', 'memory in MB',
                 "extent", "motiflet", "elbows"])

    df_single = pd.DataFrame(
        columns=['length', 'motif length', 'backend', 'time in s', 'memory in MB',
                 "extent", "motiflet", "elbows"])


    last_time = -1

    # results = []

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
                      f"Old Window {l} " +
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

                extents = dists
                motiflets = motiflets

                if subsampling:
                    print(f"\tRecomputing Extend using Window Size {l}")
                    for i, motiflet in enumerate(motiflets):
                        if motiflet:
                            motiflet = np.array(motiflet) * subsampling  # scale up again
                            preprocessing = np.array(
                                [mm.distance_preprocessing(ts_orig, l)],
                                dtype=np.float64)

                            extents[i] = get_pairwise_extent_raw(
                                ts_orig.reshape(1, -1), motiflet, l,
                                distance_single=mm.distance_single,
                                preprocessing=preprocessing)

                backend_name, new_filename = (
                    infer_filename(backend, ds_name, k_max, kwargs, subsampling))

                duration = time.time() - start
                memory_usage = mm.memory_usage

                current = [ts_orig.shape[-1],
                           l,
                           backend_name,
                           duration,
                           memory_usage,
                           extents,
                           motiflets,
                           elbow_points]
                df.loc[len(df.index)] = current
                df.to_json(new_filename + ".json")

                current_single = [ts_orig.shape[-1],
                           l,
                           backend_name,
                           duration,
                           memory_usage,
                           float(extents[-1]),
                           motiflets[-1],
                           elbow_points]
                df_single.loc[len(df_single.index)] = current_single
                df_single.to_csv(new_filename + ".csv", index=False)

                print(f"\tDiscovered motiflets in {duration:0.2f} seconds")
                print("\t'length', 'motif length', 'backend', 'time in s', "
                      "'memory in MB', 'extent', 'motiflet', 'elbows'")
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


def infer_filename(backend, ds_name, k_max, kwargs, subsampling):
    backend_name = backend
    new_filename = f"results/scalability_n_{ds_name}_{k_max}_{backend}"

    if backend == "pyattimo":
        pyattimo_delta = force_get("pyattimo_delta", kwargs)
        backend_name = f"{backend} (delta={pyattimo_delta})"

        new_filename = (new_filename +
                        f"_delta_{pyattimo_delta}")

    elif backend == "annoy":
        annoy_n_trees = force_get("annoy_n_trees", kwargs)
        annoy_search_k = force_get("annoy_search_k", kwargs)
        backend_name = \
            (f"{backend} "
             f"(annoy_n_trees={annoy_n_trees};"
             f"annoy_search_k={annoy_search_k})"
             )

        new_filename = (
                new_filename +
                f"_n_trees={annoy_n_trees}"
                f"_search_k={annoy_search_k}"
        )

    elif backend == "pynndescent":
        pynndescent_n_neighbors = force_get("pynndescent_n_neighbors", kwargs)
        pynndescent_leaf_size = force_get("pynndescent_leaf_size", kwargs)
        pynndescent_pruning_degree_multiplier = force_get(
            "pynndescent_pruning_degree_multiplier", kwargs)
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


        elif faiss_index in ["LSH"]:
            faiss_nbits = force_get("faiss_nbits", kwargs)

            backend_name = (f"{backend} "
                            f"(index={faiss_index};"
                            f"faiss_nbits={faiss_nbits})")

            new_filename = (new_filename +
                            f"_backend_{backend}"
                            f"_index_{faiss_index}"
                            f"_faiss_nbits_{faiss_nbits}")

    if subsampling:
        backend_name = f"{backend_name} (subsampling={subsampling})"
        new_filename = (new_filename + f"_subs_{subsampling}")

    return backend_name, new_filename
