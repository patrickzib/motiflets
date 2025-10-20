import warnings

from motiflets.plotting import *
from tests._datasets import read_penguin

warnings.simplefilter("ignore")


def test_input_dims():
    ds_name, series = read_penguin()

    length = 10_000

    # first test 1d np-array input
    ts_1d_numpy = series.iloc[:length, 0].values.flatten()
    ts_2d_numpy = series.iloc[:length, 0].values

    ts_pd_1d = series.iloc[:length, 0]
    ts_pd_2d = series.iloc[:length, [0]].T

    inputs = [ts_1d_numpy, ts_2d_numpy, ts_pd_1d, ts_pd_2d]
    dists, motiflets, all_elbow_points = check(ds_name, ts_pd_2d)

    for series in inputs:
        dists_new, motiflets_new, all_elbow_points_new = check(ds_name, series)

        # compare if the results are equal
        assert len(dists_new) == len(dists)
        assert len(motiflets_new) == len(motiflets)
        assert len(all_elbow_points_new) == len(all_elbow_points)

        assert (all_elbow_points_new == all_elbow_points).all()
        assert (dists_new == dists).all()



def check(ds_name, series):
    ml = Motiflets(
        ds_name,
        series,
        n_jobs=4,
        backend="default"
    )
    k_max = 10
    dists, motiflets, elbow_points = ml.fit_k_elbow(
        k_max,
        motif_length=22,
        plot_elbows=False,
        plot_motifs_as_grid=False
    )
    return dists, motiflets, elbow_points


