import pyattimo
import scipy.io as sio
from openpyxl.utils.units import points_to_pixels

from motiflets.plotting import *
from motiflets.motiflets import *
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings
warnings.simplefilter("ignore")

import logging
logging.basicConfig(level=logging.WARN)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

path = "../datasets/experiments/"

def test_plot_data():
    ds_name, series = read_penguin_data()
    series = series.iloc[497699 - 5000: 497699 + 5000, 0].T

    ml = Motiflets(ds_name, series)
    points_to_plot = 10_000
    ml.plot_dataset(max_points=points_to_plot, path="results/images/penguin_data.pdf")


def read_penguin_data():
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguin Wing-Flaps"
    return ds_name, series


def read_penguin_data_short():
    test = sio.loadmat(path + 'penguinshort.mat')
    series = pd.DataFrame(test["penguinshort"]).T
    ds_name = "Penguins (Snippet)"
    return ds_name, series

def test_plotting():
    ds_name, ts = read_penguin_data()
    ts = ts.iloc[497699 - 20_000: 497699 + 20_000, -2].T

    mm = Motiflets(ds_name, ts)
    mm.plot_dataset(path="results/images/penguin_data_raw.pdf")



def test_attimo():
    ds_name, ts = read_penguin_data()
    # ts = ts.iloc[497699 - 50_000: 497699 + 50_000, 0].T
    # ts = ts.iloc[497699 - 10_000: 497699 + 10_000, 0].T
    ts = ts.iloc[497699 - 20_000: 497699 + 20_000, 0].T

    print("Size of DS: ", ts.shape)

    start = time.time()

    l = 125 #23
    k_max = 10
    m_iter = pyattimo.MotifletsIterator(
        ts, w=l, support=k_max, top_k=1
    )

    motifs = []
    for m in m_iter:
        print(m.indices)
        print(m.extent)
        motifs.append(m.indices)
        # np.sort(m.indices)

    elbow_points = filter_unique(np.arange(len(motifs)), motifs, l)

    fig, gs = plot_motifsets(
        ds_name,
        ts,
        motifsets=np.array(motifs, dtype=np.object_)[elbow_points],
        motif_length=l,
        show=False)

    plt.savefig("results/images/penguin_pyattimo.pdf")

    end = time.time()
    print("Discovered motiflets in", end - start, "seconds")


# def test_motiflets():
#     ds_name, ts = read_penguin_data()
#     ts = ts.iloc[497699 - 50_000: 497699 + 50_000, 0].T
#
#     print("Size of DS: ", ts.shape)
#
#     l = 125 #23
#     k_max = 20
#     mm = Motiflets(ds_name, ts)
#     mm.fit_k_elbow(
#         k_max, l, plot_elbows=True,
#         plot_motifs_as_grid=True)
#
#     mm.plot_motifset(path="results/images/penguin_motiflets.pdf")
#
#     # fig, ax = plot_motifsets(
#     #    "ECG",
#     #    ts,
#     #    motifsets=motifs,
#     #    motif_length=l,
#     #    show=False)
