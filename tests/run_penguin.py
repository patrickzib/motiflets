import sys

sys.path.insert(0, "../../")
sys.path.insert(0, "../")

import scipy.io as sio

import utils as ut
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
    ds_name, series = read_penguin_1m(channel="X-Acc")
    return ds_name, series.T


def read_penguin_1m(channel=None):
    path = "../datasets/experiments/"
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguin1M"

    if channel is not None:
        return ds_name, series[[channel]]

    return ds_name, series


def read_penguin_3m():
    path = "../datasets/PeVAMmotif/"
    ds_name = "Penguin3M"
    test = sio.loadmat(path + 'penguinLabel.mat')
    series = test["data"].T
    return ds_name, pd.DataFrame(series[2, :]).T


def test_plotting():
    ds_name, ts = read_penguin_1m()
    ts = ts.iloc[497699 - 50_000: 497699 + 50_000, -2]

    mm = Motiflets(ds_name, ts)
    mm.plot_dataset(path="results/images/penguin_data_raw.pdf")


def run_motiflets_scale_n(
        backends=["scampi"],
        delta=None,
        use_1m=False,
        k_max = 10,
):
    n_range = [3 * 10^6]
    l_range = [32, 64, 23]

    for backend in backends:
        ut.test_motiflets_scale_n(
            read_penguin_1m_channel0 if use_1m else read_penguin_3m,
            n_range,
            l_range,
            k_max,
            backend=backend,
            scampi_delta=delta
        )


def main():
    print("running")
    run_motiflets_scale_n()


if __name__ == "__main__":
    main()
