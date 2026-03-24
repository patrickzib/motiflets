from motiflets.motiflets import *
from motiflets.plotting import *


def test_motiflet_top_k():
    file = 'muscle_activation.csv'
    ds_name = "Muscle Activation"
    series, df_gt = read_dataset_with_index(file)

    ml = Motiflets(ds_name, series, df_gt)
    ml.plot_dataset()

    k = 15
    length_range = np.arange(400, 701, 100)

    motif_length = ml.fit_motif_length(k, length_range)
    print("Found motif length", motif_length)

    ml.fit_k_elbow(k, motif_length, top_N=2)