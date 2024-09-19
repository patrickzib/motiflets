from datetime import datetime

from motiflets.plotting import *
from motiflets.motiflets import *

import gc
import warnings
warnings.simplefilter("ignore")

def test_motiflets_scale_n(
        read_data,
        length_range,
        l, k_max,
        backends=["default", "pyattimo", "scalable"],

        ):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    df = pd.DataFrame(columns=['length', 'backend', 'time in s', 'memory in MB', "extent"])

    results = []
    for backend in backends:
        last_n = 0
        for n in length_range:
            start = time.time()
            print(backend, n)

            ds_name, ts = read_data()
            ts = ts.iloc[:n]

            if len(ts) <= last_n:
                break

            print("Size of DS: ", ts.shape)

            mm = Motiflets(ds_name, ts, backend=backend, n_jobs=64)
            dists, _, _ = mm.fit_k_elbow(
                k_max, l, plot_elbows=False,
                plot_motifs_as_grid=False)

            duration = time.time() - start
            memory_usage = mm.memory_usage
            extent = dists[-1]

            current = [len(ts), backend, duration, memory_usage, extent]

            results.append(current)
            df.loc[len(df.index)] = current

            new_filename = f"results/scalability_n_{ds_name}_{l}_{k_max}_{timestamp}.csv"

            df.to_csv(new_filename, index=False)
            print("\tDiscovered motiflets in", duration, "seconds")
            print("\t", current)

            gc.collect()

            last_n = len(ts)

    print(results)