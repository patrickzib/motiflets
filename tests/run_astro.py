# import psutil
# import pandas as pd
import utils as ut
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
    ml.plot_dataset(
        max_points=points_to_plot,
        path="results/images/astro_data.pdf")


def run_motiflets_scale_n(
        backends=["pyattimo"],
        delta = None,
        subsampling = None
    ):
    n_range = 100_000 * np.arange(1, 200, 1)
    l_range = [70 * 38]  # roughly 6.5 seconds
    k_max = 10  # 40

    for backend in backends:
        ut.test_motiflets_scale_n(
            read_data,
            n_range,
            l_range,
            k_max,
            backend,
            pyattimo_delta=delta,
            subsampling=subsampling
        )


def main():
    print("running")
    run_motiflets_scale_n()

if __name__ == "__main__":
    main()
