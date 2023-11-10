import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 150

path = "../../motiflets_use_cases/audio/"

datasets = {
    "Numb - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": np.arange(2.5, 4.5, 0.1),
        "ds_name": "Numb - Linkin Park",
        "audio_file_url": path + "Numb - Linkin Park.wav",
        "lrc_url": path + "Numb - Linkin Park.lrc"
    },
    "What I've Done - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": np.arange(6.0, 8, 0.1),  # 7.75,
        "ds_name": "What I've Done - Linkin Park",
        "audio_file_url": path + "What I've Done - Linkin Park.wav",
        "lrc_url": path + "What I've Done - Linkin Park.lrc"
    },
    "The Rolling Stones - Paint It, Black": {
        "ks": 15,
        "channels": 3,
        "length_in_seconds": np.arange(5.0, 6.0, 0.1),
        "ds_name": "The Rolling Stones - Paint It, Black",
        "audio_file_url": path + "The Rolling Stones - Paint It, Black.wav",
        "lrc_url": path + "The Rolling Stones - Paint It, Black.lrc"
    }
}

dataset = datasets["The Rolling Stones - Paint It, Black"]
# dataset = datasets["What I've Done - Linkin Park"]
k_max = dataset["ks"]
n_dims = dataset["channels"]
motif_length_range_in_s = dataset["length_in_seconds"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
lrc_url = dataset["lrc_url"]


def read_songs():
    # Read audio from wav file
    x, sr, mfcc_f, audio_length_seconds = read_audio(audio_file_url, True)
    print("Length:", audio_length_seconds)
    index_range = np.arange(0, mfcc_f.shape[1]) * audio_length_seconds / mfcc_f.shape[1]
    df = pd.DataFrame(mfcc_f,
                      index=["MFCC " + str(a) for a in np.arange(0, mfcc_f.shape[0])],
                      columns=index_range
                      )
    df.index.name = "MFCC"
    return audio_length_seconds, df, index_range

def test_audio():
    audio_length_seconds, df, index_range = read_songs()
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6']
    df = df.loc[channels]

    subtitles = read_lrc(lrc_url)
    df_sub = get_dataframe_from_subtitle_object(subtitles)
    df_sub.set_index("seconds", inplace=True)

    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   dimension_labels=df.index,
                   n_dims=n_dims,
                   )

    motif_length, _ = ml.fit_motif_length(
        k_max,
        motif_length_range,
        plot=True,
        plot_elbows=True,
        plot_motifsets=True,
        plot_best_only=False
    )

    length_in_seconds = motif_length * audio_length_seconds / df.shape[1]
    print("Found motif length", length_in_seconds, motif_length)

    # best motiflets
    for a, eb in enumerate(ml.elbow_points):
        motiflet = np.sort(ml.motiflets[eb])
        print("Positions:", index_range[motiflet])

        lyrics = []
        for i, m in enumerate(motiflet):
            l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
            lyrics.append(l)
            print(i + 1, l)

        # path_ = "audio/snippets/" + ds_name + "_Dims_" + str(n_dims) + "_Motif.pdf"
        # ml.plot_motifset(path=path_)

        extract_audio_segment(
            df, ds_name, audio_file_url, "snippets",
            length_in_seconds, index_range, motif_length, motiflet, id=(a+1))