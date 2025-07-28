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
    k_max = 9
    motif_length = 27

    lengths = [
        1_000,
        5_000,
        10_000,
        30_000,
        50_000
    ]

    ds_name, B = read_penguin_data()

    for i, length in enumerate(lengths):
        series = B.iloc[:length, 0].values
        print("-------------------------------\n")
        print(f"Current length: {series.shape}")

        t_before = time.perf_counter()
        ml = Motiflets(
            ds_name,
            series,
            n_jobs=-1,
            backend="default"
        )

        gt_dists, gt_motiflets, gt_elbow_points = ml.fit_k_elbow(
            k_max=k_max,
            motif_length=motif_length,
            plot_elbows=False,
            plot_motifs_as_grid=False
        )

        print(f"Testing backend: 'default'")
        print(f"\tRuntime: {time.perf_counter() - t_before:0.1f} s")
        print(f"\tMemory usage: {ml.memory_usage:0.1f} MB")
        print("\tElbow points:", gt_elbow_points)
        print("\tMotiflets:", *gt_motiflets[gt_elbow_points])
        print("\tDistances:", *gt_dists[gt_elbow_points])

        del ml

        for backend in ["scalable", "sparse"]:
            print(f"Testing backend: {backend}")

            ml = Motiflets(
                ds_name,
                series,
                n_jobs=-1,
                backend=backend
            )

            t_before = time.perf_counter()
            dists, motiflets, elbow_points = ml.fit_k_elbow(
                k_max=k_max,
                motif_length=motif_length,
                plot_elbows=False,
                plot_motifs_as_grid=False
            )

            print(f"\tRuntime: {time.perf_counter() - t_before:0.1f} s")
            print(f"\tMemory usage: {ml.memory_usage:0.1f} MB")
            print("\tElbow points:", elbow_points)
            print("\tMotiflets:", *motiflets[elbow_points])
            print("\tDistances:", *dists[elbow_points])

            assert np.allclose(gt_elbow_points, elbow_points, rtol=1e-2, atol=1e-2), \
                f"Elbow points do not match for backend {backend} with length {length}"

            for elbow in elbow_points:
                assert np.allclose(np.sort(gt_motiflets[elbow]),
                                   np.sort(motiflets[elbow])), \
                    f"Motiflets do not match for {backend} with length {length}"
                assert np.allclose(gt_dists[elbow],
                                   dists[elbow], rtol=1e-2, atol=1e-2), \
                    f"Distances do not match for {backend} with length {length}"

            print(f"Backend {backend} passed for series length {length}.\n")

            del ml

        print("-------------------------------\n")
