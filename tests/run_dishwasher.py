import sys

sys.path.insert(0, "../../")
sys.path.insert(0, "../")

import utils as ut
from motiflets.motiflets import *
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

path = "../datasets/original/"

def read_data():
    file = 'dishwasher.txt'
    ds_name = "Dishwasher"
    series = pd.read_csv(path+file, header=None).squeeze('columns')
    print(f"Loaded dataset {ds_name} with length {len(series)}")
    return ds_name, series

def test_plot_data():
    ds_name, series = read_data()
    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(
        max_points=points_to_plot,
        path="results/images/dishwasher_data.pdf")


def run_motiflets_scale_n(
        backends=["scampi"],
        delta=None,
        k_max = 10,
    ):
    n_range = [245_152]
    l_range = [512, 1024, 2048, 4096]


    for backend in backends:
        ut.test_motiflets_scale_n(
            read_data,
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
