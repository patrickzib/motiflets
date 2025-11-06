# import psutil
# import pandas as pd
from datetime import datetime

import scipy.io as sio

import utils as ut
from motiflets.motiflets import *
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150


def test_plot_data():
    ds_name, series = read_penguin_1m()
    series = series.iloc[497699 - 5000: 497699 + 5000, 0].T

    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(
        max_points=points_to_plot,
        path="results/images/penguin_data.pdf")


def read_penguin_1m_channel0():
    return read_penguin_1m(channel="X-Acc")


def read_penguin_1m(channel=None):
    path = "../datasets/experiments/"
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguin1M"

    if channel is not None:
        return ds_name, series[channel]

    return ds_name, series


def read_penguin_3m():
    path = "../datasets/PeVAMmotif/"
    ds_name = "Penguin3M"
    test = sio.loadmat(path + 'penguinLabel.mat')
    series = test["data"].T
    return ds_name, series[2, :]


def test_plotting():
    ds_name, ts = read_penguin_1m()
    ts = ts.iloc[497699 - 50_000: 497699 + 50_000, -2].T

    mm = Motiflets(ds_name, ts)
    mm.plot_dataset(path="results/images/penguin_data_raw.pdf")


def run_motiflets_scale_n(
        backends=["pyattimo"],
        delta=None,
        subsampling=None,
        use_1m=True
):
    n_range = 100_000 * np.arange(1, 200, 1)
    l_range = [125]  # 23
    k_max = 10  # 20

    for backend in backends:
        ut.test_motiflets_scale_n(
            read_penguin_1m_channel0 if use_1m else read_penguin_3m,
            n_range,
            l_range,
            k_max,
            backend,
            pyattimo_delta=delta,
            subsampling=subsampling
        )


# def test_motiflets_scale_k():
#     timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
#
#     df = pd.DataFrame(columns=['length', 'backend', 'time in s', 'memory in MB'])
#
#     results = []
#
#     n = 25_000
#     k_range = np.arange(5, 45, 5)
#     for k_max in k_range:
#         for backend in ["default", "scalable", "pyattimo", "extent"]:
#             start = time.time()
#             print(backend, k_max)
#             ds_name, ts = read_penguin_1m()
#             ts = ts.iloc[497699 - n: 497699 + n, 0].T
#             print("Size of DS: ", ts.shape)
#
#             l = 125  # 23
#             mm = Motiflets(ds_name, ts, backend=backend, n_jobs=64)
#             dists, _, _ = mm.fit_k_elbow(
#                 k_max, l, plot_elbows=True,
#                 plot_motifs_as_grid=True)
#
#             duration = time.time() - start
#             memory_usage = mm.memory_usage
#             extent = dists[-1]
#
#             current = [k_max, backend, duration, memory_usage, extent]
#
#             results.append(current)
#             df.loc[len(df.index)] = current
#
#             new_filename = f"results/scalability_k_{timestamp}.csv"
#
#             df.to_csv(new_filename, index=False)
#             print("\tDiscovered motiflets in", duration, "seconds")
#             print("\t", current)
#
#     print(results)
#
#     # mm.plot_motifset(path="results/images/penguin_motiflets.pdf")
#     # fig, ax = plot_motifsets(
#     #    "ECG",
#     #    ts,
#     #    motifsets=motifs,
#     #    motif_length=l,
#     #    show=False)


def main():
    print("running")
    run_motiflets_scale_n()


if __name__ == "__main__":
    main()
