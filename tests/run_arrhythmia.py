import utils as ut
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
path = "../datasets/experiments/"


def read_data():
    series = pd.read_csv(path + "arrhythmia_subject231_channel0.csv")
    ds_name = "Arrhythmia"
    return ds_name, series.iloc[:, 0].T


def test_plot_data():
    ds_name, series = read_data()
    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(
        max_points=points_to_plot,
        path="results/images/arrhythmia_data.pdf")


def run_motiflets_scale_n(
        backends=["pyattimo"],
        delta=None,
        subsampling=None
):
    n_range = 100_000 * np.arange(1, 200, 1)
    l_range = [200]
    k_max = 10  # 20

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
