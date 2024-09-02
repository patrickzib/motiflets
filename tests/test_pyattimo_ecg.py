import pyattimo

from motiflets.plotting import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import warnings

warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 150


def test_attimo():
    import logging
    logging.basicConfig(level=logging.WARN)

    # load a dataset, any list of numpy array of floats works fine
    # The following call loads the first 100000 points of the ECG
    # dataset (which will be downloaded from the internet)
    ts = pyattimo.load_dataset('ecg', 100_000)
    print("Size of DS: ", ts.shape)

    # Now we can find k-motiflets:
    #  - w is the window length
    #  - support is the number of subsequences in the motiflet (k in the motiflet paper)
    #  - repetitions is the number of LSH repetitions
    start = time.time()

    l = 1000
    k_max = 10
    m_iter = pyattimo.MotifletsIterator(
        ts, w=l, support=k_max
    )

    motifs = []
    for m in m_iter:
        print(m.indices)
        print(m.extent)
        motifs.append(m.indices)
        # np.sort(m.indices)

    _ = plot_motifsets(
        "ECG",
        ts,
        motifsets=motifs,
        motif_length=l,
        show=False)

    plt.savefig("results/images/ecg_pyattimo.pdf")

    end = time.time()
    print("Discovered motiflets in", end - start, "seconds")


def test_motiflets():
    # load a dataset, any list of numpy array of floats works fine
    # The following call loads the first 100000 points of the ECG
    # dataset (which will be downloaded from the internet)
    ts = pyattimo.load_dataset('ecg', 100_000).flatten()
    print("Size of DS: ", ts.shape)

    k = 20
    l = 1000
    mm = Motiflets("ECG", ts)
    mm.fit_k_elbow(
        k, l, plot_elbows=True,
        plot_motifs_as_grid=True)

    mm.plot_motifset(path="results/images/ecg_motiflets.pdf")

    # fig, ax = plot_motifsets(
    #    "ECG",
    #    ts,
    #    motifsets=motifs,
    #    motif_length=l,
    #    show=False)
