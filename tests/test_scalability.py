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
    lengths = [
        1_000,
        5_000,
        10_000,
        30_000,
        50_000
        # 100_000,
        # 150_000,
        # 200_000,
        # 250_000
    ]

    ds_name, B = read_penguin_data()
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("Current", length)
        series = B.iloc[:length, 0].T.values

        ml = Motiflets(
            ds_name,
            series,
            n_jobs=8,
            backend="sparse"
        )

        k_max = 10

        t_before = time.time()
        dists, motiflets, elbow_points = ml.fit_k_elbow(
            k_max,
            22,
            plot_elbows=False,
            plot_motifs_as_grid=False
        )
        t_after = time.time()
        time_s[i] = t_after - t_before
        print("\tTotal Time:", time_s[i])
        print("\tElbow points:", elbow_points)
        print("\tMotiflets:", motiflets)
        print("\tDistances:", dists)

        #dict = time_s
        #df = pd.DataFrame(data=dict, columns=['Time'], index=lengths)
        #df["Method"] = "Motiflets (one-dim)"
        #df.index.name = "Lengths"
        # df.to_csv('csv/scalability_univ_motiflets_k5.csv')