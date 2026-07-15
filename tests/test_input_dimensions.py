import warnings

import numpy as np
import pandas as pd

from motiflets import Motiflets
from motiflets.motiflets import compute_distances_full, flatten_elbows
from motiflets.plotting import *
from motiflets.plotting import _scale_motifset_positions
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


def test_compute_distances_full_accepts_1d_input():
    _, series = read_penguin()
    data_1d = series.iloc[:128, 0]
    data_2d = data_1d.to_frame().T

    D_1d = compute_distances_full(data_1d, m=16)
    D_2d = compute_distances_full(data_2d, m=16)

    assert D_1d.shape == D_2d.shape
    assert (D_1d == D_2d).all()


def test_fit_k_elbow_returns_all_computed_candidates():
    ds_name, series = read_penguin()
    ml = Motiflets(
        ds_name,
        series.iloc[:300, 0],
        n_jobs=2,
        backend="default"
    )

    dists, motiflets, elbow_points = ml.fit_k_elbow(
        k_max=6,
        motif_length=20,
        plot_elbows=False,
        plot_motifs_as_grid=False
    )

    assert len(dists) == len(ml.dists)
    assert len(motiflets) == len(ml.motiflets)
    assert elbow_points.tolist() == ml.elbow_points[0].tolist()
    assert motiflets[2] is not None
    assert motiflets[3] is not None
    assert np.isfinite(dists[2])
    assert np.isfinite(dists[3])


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
