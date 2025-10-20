import numpy as np
import warnings

from motiflets.plotting import *
from tests._datasets import read_penguin

warnings.simplefilter("ignore")

np.printoptions(precision=2, suppress=True)


def test_motiflets_univariate():
    k_max = 10
    window_size = 10

    ds_name, B = read_penguin()
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
