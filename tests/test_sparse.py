import warnings

from motiflets.plotting import *

warnings.simplefilter("ignore")

path = "../datasets/experiments/"

def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"
    return ds_name, series


def test_motiflets():
    ds_name, T = read_penguin_data()
    n = 2_000
    series = T.iloc[497699:497699 + n, [0]].T.to_numpy()
    ml = Motiflets(ds_name, series, backend="sparse")
    ks = 20
    motif_length = 22
    _ = ml.fit_k_elbow(ks, motif_length, plot_motifs_as_grid=False)


def test_motiflets_scalable():
    ds_name, T = read_penguin_data()
    n = 10_000
    series = T.iloc[497699:497699 + n, [0, 1, 2]].T.to_numpy()

    ml = Motiflets(ds_name, series, backend="sparse")
    ks = 20
    motif_length = 22
    _ = ml.fit_k_elbow(ks, motif_length, plot_motifs_as_grid=False)


def test_motiflets_default():
    ds_name, T = read_penguin_data()
    n = 10_000
    series = T.iloc[497699:497699 + n, [0, 1, 2]].T.to_numpy()

    ml = Motiflets(ds_name, series, backend="default")
    ks = 20
    motif_length = 22
    _ = ml.fit_k_elbow(ks, motif_length, plot_motifs_as_grid=False)
