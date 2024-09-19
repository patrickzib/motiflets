import utils as ut
from motiflets.motiflets import *
from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import logging
logging.basicConfig(level=logging.WARN)
pyattimo_logger = logging.getLogger('pyattimo')
pyattimo_logger.setLevel(logging.WARNING)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"


def find_dominant_window_sizes(X, offset=0.05):
    """Determine the Window-Size using dominant FFT-frequencies.

    Parameters
    ----------
    X : array-like, shape=[n]
        a single univariate time series of length n
    offset : float
        Exclusion Radius

    Returns
    -------
    trivial_match: bool
        If the candidate change point is a trivial match
    """
    fourier = np.absolute(np.fft.fft(X))
    freqs = np.fft.fftfreq(X.shape[0], 1)

    coefs = []
    window_sizes = []

    for coef, freq in zip(fourier, freqs):
        if coef and freq > 0:
            coefs.append(coef)
            window_sizes.append(1 / freq)

    coefs = np.array(coefs)
    window_sizes = np.asarray(window_sizes, dtype=np.int64)

    idx = np.argsort(coefs)[::-1]
    return next(
        (
            int(window_size / 2)
            for window_size in window_sizes[idx]
            if window_size in range(20, int(X.shape[0] * offset))
        ),
        window_sizes[idx[0]],
    )


def load_dataset(selection=None):
    desc_filename = path + "pamap_desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []
    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue

        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        ts = np.load(file=path + "pamap_data.npz")[ts_name]

        df.append(
            (ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return "PAMAP", pd.DataFrame.from_records(
        df, columns=["name", "window_size", "change_points", "time_series"])


def test_plot_data():
    selection = [126]  # Outdoor

    ds_name, series = load_dataset(selection)
    ts = series.time_series[0]

    ml = Motiflets(ds_name, ts)
    points_to_plot = 10_000
    ml.plot_dataset(max_points=points_to_plot,
                    path="results/images/pamap_data.pdf")


def test_attimo():
    ds_name, series = load_dataset()
    ts = series.time_series[0]

    # l = 2 * find_dominant_window_sizes(ts, offset=0.05)
    l = 200

    print("Size of DS: ", ts.shape, " l:", l)
    start = time.time()

    k_max = 10
    m_iter = pyattimo.MotifletsIterator(
        ts, w=l, support=k_max, top_k=1
    )

    motifs = []
    for m in m_iter:
        print(m.indices)
        print(m.extent)
        motifs.append(m.indices)

    elbow_points = filter_unique(np.arange(len(motifs)), motifs, l)

    points_to_plot = 10_000
    _ = plot_motifsets(
        ds_name,
        ts,
        max_points=points_to_plot,
        motifsets=np.array(motifs, dtype=np.object_)[elbow_points],
        motif_length=l,
        show=False)

    end = time.time()

    # plt.savefig("results/images/pamap_pyattimo.pdf")

    print("Discovered motiflets in", end - start, "seconds")


def test_motiflets():
    ds_name, series = load_dataset()
    ts = series.time_series[0]

    # l = 2 * find_dominant_window_sizes(ts, offset=0.05)
    l = 200

    print("Size of DS: ", ts.shape, " l:", l)
    start = time.time()

    k_max = 10

    mm = Motiflets(ds_name, ts, backend="scalable", n_jobs=8)
    mm.fit_k_elbow(
        k_max, l, plot_elbows=False,
        plot_motifs_as_grid=False)

    end = time.time()

    # plt.savefig("results/images/pamap_pyattimo.pdf")

    print("Discovered motiflets in", end - start, "seconds")

    # mm.plot_motifset(path="results/images/penguin_motiflets.pdf")

    # fig, ax = plot_motifsets(
    #    "ECG",
    #    ts,
    #    motifsets=motifs,
    #    motif_length=l,
    #    show=False)


def test_motiflets_sparse():
    lengths = [1_000,
               5_000,
               10_000,
               30_000,
               # 50_000,
               # 100_000,
               # 150_000,
               # 200_000,
               # 250_000
               ]

    ds_name, series = load_dataset()
    B = series.time_series[0]
    time_s = np.zeros(len(lengths))

    for i, length in enumerate(lengths):
        print("--------------------")
        print("Current", length)
        for distance in ["znormed_ed", "ed", "cosine"]:
            series = B[:length]

            print("Distance", distance)
            ml = Motiflets(
                ds_name, series, distance=distance,
                n_jobs=8, backend="scalable"
            )

            k_max = 20

            t_before = time.time()
            _ = ml.fit_k_elbow(
                k_max,
                22,
                plot_elbows=False,
                plot_motifs_as_grid=False
            )
            t_after = time.time()
            time_s[i] = t_after - t_before

            memory_usage = ml.memory_usage

            print("Time:", time_s[i], "s")
            print("Memory:", memory_usage, "MB")


def test_motiflets_scale_n():
    length_range = 10_000 * np.arange(1, 200, 1)
    l = 200
    k_max = 10
    backends = ["default", "pyattimo", "scalable"]

    ut.test_motiflets_scale_n(
        load_dataset,
        length_range,
        l, k_max,
        backends
    )


def main():
    print("running")
    test_motiflets_scale_n()

if __name__ == "__main__":
    main()