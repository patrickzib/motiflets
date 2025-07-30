import warnings

from motiflets.plotting import *

warnings.simplefilter("ignore")

path = "../datasets/experiments/"

import numpy as np

np.printoptions(precision=2, suppress=True)


def read_penguin_data():
    series = pd.read_csv(
        path + "penguin.txt",
        names=(["X-Acc", "Y-Acc", "Z-Acc", "4", "5", "6", "7", "Pressure", "9"]),
        delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


def test_motiflets_univariate():
    k_max = 10
    window_size = 10

    ds_name, B = read_penguin_data()
    data = B.iloc[:1111, 0].values

    print(f"Trimming to {len(data)} {data.dtype}")

    D_knns, knns = ml.compute_distances_with_knns(
        data.reshape(1, -1),
        m=window_size,
        k=k_max,
        n_jobs=1
    )

    for jobs in [2, 4, 8]:
        print(f"Testing {jobs} jobs")
        D_knns2, knns2 = ml.compute_distances_with_knns(
            data.reshape(1, -1),
            m=window_size,
            k=k_max,
            n_jobs=jobs
        )

        assert (np.allclose(D_knns, D_knns2)), "D_knns do not match"