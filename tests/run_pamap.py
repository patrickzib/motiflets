import matplotlib as mpl

import utils as ut
from motiflets.plotting import *

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"


def read_data(selection=None):
    desc_filename = path + "pamap_desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []
    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue

        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        ts = np.load(file=path + "pamap_data.npz")[ts_name]

        df.append(
            (ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return "PAMAP", pd.DataFrame.from_records(
        df, columns=["name", "window_size", "change_points", "time_series"]).time_series[0]


def test_plot_data():
    selection = [126]  # Outdoor

    ds_name, series = read_data(selection)
    ts = series.time_series[0]

    ml = Motiflets(ds_name, ts)
    points_to_plot = 10_000
    ml.plot_dataset(
        max_points=points_to_plot,
        path="results/images/pamap_data.pdf")


def run_motiflets_scale_n(
        backends=["pyattimo"],
        delta = None,
        subsampling=None
    ):
    n_range = 50_000 * np.arange(1, 200, 1)
    l_range = [200]
    k_max = 10

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