# import psutil
# import pandas as pd
import utils as ut
# from motiflets.motiflets import *
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import logging

logging.basicConfig(level=logging.WARN)
pyattimo_logger = logging.getLogger('pyattimo')
pyattimo_logger.setLevel(logging.WARNING)

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/original/"


def read_eeg_data():
    file = 'npo141.csv'  # Dataset Length n:  269286
    ds_name = "EEG-Sleep"
    series = pd.read_csv(path + file, header=None).squeeze('columns')
    return ds_name, series


def test_plot_data():
    ds_name, series = read_eeg_data()
    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(max_points=points_to_plot, path="results/images/eeg_data.pdf")


def test_motiflets_scale_n(
        backends = ["default", "pyattimo", "scalable"],
        delta = None,
        subsampling = None
):
    length_range = 50_000 * np.arange(1, 200, 1)

    l = 25 * 25  # roughly 6.5 seconds
    k_max = 10  # 20

    ut.test_motiflets_scale_n(
        read_eeg_data,
        length_range,
        [l], k_max,
        backends,
        delta=delta,
        subsampling=subsampling
    )


def main():
    print("running")
    test_motiflets_scale_n()


if __name__ == "__main__":
    main()
