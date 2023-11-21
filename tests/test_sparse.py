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

    k = 20
    m = 22
    _ = ml.fit_k_elbow(k, m)


def test_motiflets_sparse():
    ds_name, T = read_penguin_data()
    n = 40_001
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    ml = Motiflets(ds_name, series)
    m = 100
    k = 20
    _ = ml.fit_k_elbow(k, m)


def test_sparse_matrix():
    ds_name, T = read_penguin_data()
    n = 40_001
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    m = 100
    k = 20
    D_sparse, knns = ml.compute_distances_with_knns_sparse(series, m=m, k=k)

    elements = 0
    for A in D_sparse:
        elements += len(A)

    n = (series.shape[0] - m + 1)
    print("Total:", elements, n ** 2, str(elements * 100 / n ** 2) + "%")

    non_empty = 0
    for A in D_sparse:
        non_empty += len(A) > k

    print("Non-Empty:", non_empty, str(non_empty/len(D_sparse)*100) + "%")


def test_full_matrix():
    ds_name, T = read_penguin_data()
    n = 40_000
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    m = 1000
    k = 10
    _, _ = ml.compute_distances_with_knns(series, m=m, k=k)
