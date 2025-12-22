from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t",
                         header=None,
                         dtype=np.float64)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


def test_motiflets_sparse():
    lengths = [
        #1_000,
        #5_000,
        #10_000,
        #30_000,
        #50_000,
        #100_000,
        150_000,
        #200_000,
        #250_000
    ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("--------------------")
        for distance in ["znormed_ed"]:
            for backend in ["scalable"]:
                series = B.iloc[:length, 0].T

                print("Distance", distance)
                ml = Motiflets(
                    ds_name,
                    series,
                    distance=distance,
                    n_jobs=-1,
                    backend=backend
                )

                k_max = 10

                t_before = time.time()
                extent, motiflets, _ = ml.fit_k_elbow(
                    k_max,
                    22,
                    plot_elbows=False,
                    plot_motifs_as_grid=False
                )

                ml.plot_motifset()

                t_after = time.time()
                time_s[i] = t_after - t_before
                memory_usage = ml.memory_usage

                print("Backend:", backend)

                print("Time:", time_s[i], "s")
                print("Memory:", memory_usage, "MB")
                print("Extent", extent[2:7])