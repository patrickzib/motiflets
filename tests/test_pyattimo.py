import pyattimo
from motiflets.plotting import *


def test_attimo():
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

    m_iter = pyattimo.MotifletsIterator(
        ts, w=1000, max_k=30
    )

    for m in m_iter:
        print(m.indices)
        print(m.extent)
        # np.sort(m.indices)

    end = time.time()
    print("Discovered motiflets in", end - start, "seconds")


def test_motiflets():
    # load a dataset, any list of numpy array of floats works fine
    # The following call loads the first 100000 points of the ECG
    # dataset (which will be downloaded from the internet)
    ts = pyattimo.load_dataset('ecg').flatten()
    print("Size of DS: ", ts.shape)

    k = 10
    mot = Motiflets("ECG", ts)
    _ = mot.fit_k_elbow(k, 1000)
