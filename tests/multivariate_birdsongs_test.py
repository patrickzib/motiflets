import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

from audio.lyrics import *


path = "../../motiflets_use_cases/birds/"

datasets = {
    "Common-Starling": {
        "ks": 5,
        "channels": 10,
        "length_range": np.arange(25, 100, 5),
        "ds_name": "Common-Starling",
        "audio_file_url": path + "xc27154---common-starling---sturnus-vulgaris.mp3",
    },
    "House-Sparrow": {
        "ks": 20,
        "channels": 10,
        "length_range": np.arange(25, 50, 5),
        "ds_name": "House-Sparrow",
        "audio_file_url": path + "house-sparrow-passer-domesticus-audio.mp3"
    }
}

dataset = datasets["House-Sparrow"]
k_max = dataset["ks"]
channels = dataset["channels"]
length_range = dataset["length_range"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]


def test_audio():
    seconds, df, index_range = read_mp3(audio_file_url)

    ml = Motiflets(ds_name, df,
                   slack=1.0,
                   dimension_labels=df.index,
                   n_dims=2,
                   )

    motif_length, all_minima = ml.fit_motif_length(
        k_max, length_range,
        plot_motifsets=False
    )
    length_in_seconds = index_range[motif_length]
    print("Best length", motif_length, length_in_seconds, "s")

    # length_in_seconds = 2.2
    # motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    # print(motif_length)

    dists, motiflets, elbow_points = ml.fit_k_elbow(
        k_max, motif_length=motif_length,
        plot_elbows=True,
        plot_motifs_as_grid=False
    )

    path_ = ("audio/bird_songs/" + ds_name +
             "_Channels_" + str(len(df.index)) +
             "_full.pdf")
    # ml.plot_dataset(path_)

    # best motiflet
    motiflet = np.sort(motiflets[elbow_points[-1]])
    print("Positions:", index_range[motiflet])

    ml.plot_motifset()

    plt.savefig(
        "audio/bird_songs/" + ds_name + "_Channels_" + str(
            len(df.index)) + "_Motif.pdf")
    plt.show()

    extract_audio_segment(
        df, ds_name, audio_file_url, "bird_songs",
        length_in_seconds, index_range, motif_length, motiflet)
