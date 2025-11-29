import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import utils as ut
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/original/"


def read_data():
    file = 'npo141.csv'  # Dataset Length n:  269286
    ds_name = "EEG-Sleep"
    series = pd.read_csv(path + file, header=None).squeeze('columns')
    print(f"Loaded dataset {ds_name} with length {len(series)}")
    return ds_name, series


def test_plot_data():
    ds_name, series = read_data()
    ml = Motiflets(
        ds_name, series, backend="pyattimo"
    )
    points_to_plot = 10_000
    ml.plot_dataset(
        max_points=points_to_plot,
        path="results/images/eeg_data.pdf")


def run_motiflets_scale_n(
        backends=["pyattimo"],
):
    n_range = [300_000]
    l_range = [25 * 25]  # roughly 6.5 seconds
    k_max = 10  # 20

    for backend in backends:
        ut.test_motiflets_scale_n(
            read_data,
            n_range,
            l_range,
            k_max,
            backend=backend
        )


def main():
    print("running")
    run_motiflets_scale_n()


if __name__ == "__main__":
    main()
