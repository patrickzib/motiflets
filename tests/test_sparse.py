import scipy.io as sio
from motiflets.plotting import *

path = "../datasets/experiments/"

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    return ds_name, series


def read_penguin_data_short():
    test = sio.loadmat(path + 'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T
    ds_name = "Penguins (Snippet)"
    return ds_name, series


def test_motiflets():
    ds_name, T = read_penguin_data()
    n = 2_000
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    ml = Motiflets(ds_name, series)

    ks = 20
    motif_length = 22
    _ = ml.fit_k_elbow(ks, motif_length)


def test_motiflets_sparse():
    ds_name, T = read_penguin_data()
    n = 100_000
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    ml = Motiflets(ds_name, series)
    ks = 20
    motif_length = 100
    _ = ml.fit_k_elbow(ks, motif_length)


def test_full_matrix():
    ds_name, T = read_penguin_data()
    n = 20_000
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    m = 1000
    k = 10
    _, _ = ml.compute_distances_with_knns(series, m=m, k=k)
