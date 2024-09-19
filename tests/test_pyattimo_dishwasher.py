import pyattimo
import scipy.io as sio
# import psutil
# import pandas as pd
from datetime import datetime

from motiflets.plotting import *
from motiflets.motiflets import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import gc
import warnings
warnings.simplefilter("ignore")

import logging
logging.basicConfig(level=logging.WARN)
pyattimo_logger = logging.getLogger('pyattimo')
pyattimo_logger.setLevel(logging.WARNING)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

path = "../datasets/original/"

def read_dishwasher():
    file = 'dishwasher.txt'  # Dataset Length n:  269286
    ds_name = "Dishwasher"
    series = pd.read_csv(path+file, header=None).squeeze('columns')
    return ds_name, series

def test_plot_data():
    ds_name, series = read_dishwasher()
    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(max_points=points_to_plot, path="results/images/dishwasher_data.pdf")


def test_motiflets_scale_n():
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    df = pd.DataFrame(columns=['length', 'backend', 'time in s', 'memory in MB', "extent"])

    results = []
    length_range = 25_000 * np.arange(20, 200, 1)
    for backend in ["pyattimo"]: #"default", "pyattimo", "scalable"
        last_n = 0
        for n in length_range:
            start = time.time()
            print(backend, n)

            ds_name, ts = read_dishwasher()
            ts = ts.iloc[:n]

            if len(ts) <= last_n:
                break

            print("Size of DS: ", ts.shape)

            l = 125*8  # roughly 6.5 seconds
            k_max = 20 # 40
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



def main():
    print("running")
    test_motiflets_scale_n()

if __name__ == "__main__":
    main()
