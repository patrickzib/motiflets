import gc
import utils as ut
from datetime import datetime

from motiflets.plotting import *
from motiflets.motiflets import *

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


def read_arrhythmia():
    series = pd.read_csv(path + "arrhythmia_subject231_channel0.csv")
    ds_name = "Arrhythmia"
    return ds_name, series.iloc[:, 0].T


def test_plot_data():
    ds_name, series = read_arrhythmia()
    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(max_points=points_to_plot,
                    path="results/images/arrhythmia_data.pdf")


def test_attimo():
    ds_name, ts = read_arrhythmia()
    # l = 2 * find_dominant_window_sizes(ts, offset=0.05)
    l = 200

    print("Size of DS: ", ts.shape, " l:", l)
    start = time.time()

    k_max = 20
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
    fig, gs = plot_motifsets(
        ds_name,
        ts,
        max_points=points_to_plot,
        motifsets=np.array(motifs, dtype=np.object_)[elbow_points],
        motif_length=l,
        show=False)

    plt.savefig("results/images/arrhythmia_pyattimo.pdf")

    end = time.time()
    print("Discovered motiflets in", end - start, "seconds")



def test_motiflets_scale_n(
        backends=["default", "pyattimo", "scalable"],
        delta=None,
        subsampling=None
):
    length_range = 50_000 * np.arange(1, 200, 1)
    l = 200
    k_max = 10 # 20

    ut.test_motiflets_scale_n(
        read_arrhythmia,
        length_range,
        l, k_max,
        backends,
        delta,
        subsampling
    )

def main():
    print("running")
    test_motiflets_scale_n()

if __name__ == "__main__":
    main()