import pyattimo
import scipy.io as sio
# import psutil
# import pandas as pd
from datetime import datetime
import utils as ut

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

def read_data():
    file = 'ASTRO.csv'  # Dataset Length n:  269286
    ds_name = "ASTRO"
    series = pd.read_csv(path+file, header=None).squeeze('columns')
    return ds_name, series

def test_plot_data():
    ds_name, series = read_data()
    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(max_points=points_to_plot, path="results/images/astro_data.pdf")


def test_motiflets_scale_n():
    length_range = 25_000 * np.arange(1, 200, 1)
    l = 70 * 38  # roughly 6.5 seconds
    k_max = 10  # 40
    backends = ["default", "pyattimo", "scalable"]

    ut.test_motiflets_scale_n(
        read_data,
        length_range,
        l, k_max,
        backends
    )


def main():
    print("running")
    test_motiflets_scale_n()

if __name__ == "__main__":
    main()