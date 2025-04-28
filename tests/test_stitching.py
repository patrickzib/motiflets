from motiflets.plotting import *
from motiflets.motiflets import *

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


def test_discover_motiflets():
    ds_name, T = read_penguin_data()
    n = 30_000
    series = T.iloc[497699:497699 + n, 0].T.to_numpy()

    ml = Motiflets(ds_name, series, backend="default")

    ks = 20
    motif_length = 100
    motif_distances, motif_candidates, _ \
        = ml.fit_k_elbow(
            ks, motif_length,
            plot_elbows=False,
            plot_motifs_as_grid=False)

    motiflet = motif_candidates[-1]
    extent = motif_distances[-1]

    new_motiflet, new_extent = stitch_and_refine(
        series,
        motif_length,
        motiflet,
        extent,
        search_window=min(4*motif_length, 1024),
        upper_bound=extent
    )
    print(f"Original motiflet: {motiflet}, size {len(motiflet)} with extent {extent}")
    print(f"Refined  motiflet: {new_motiflet}, size {len(new_motiflet)} with extent {new_extent}")
