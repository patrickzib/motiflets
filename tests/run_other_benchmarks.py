import traceback

import run_gap as gap
import run_pamap as pamap
import run_penguin as penguin
import run_astro as astro
import run_dishwasher as dishwasher
import run_eeg_physiodata as eeg
import run_arrhythmia as arrhythmia


def run_safe(module, backends, delta, use_1m=None):
    try:
        if use_1m is not None:
            module.run_motiflets_scale_n(backends=backends, delta=delta, use_1m=use_1m)
        else:
            module.run_motiflets_scale_n(backends=backends, delta=delta)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


def main():
    # backends = ["pyattimo"]
    # deltas = [None, 0.25, 0.50]
    #
    # for delta in deltas:
    #     print(f"Using delta {delta}")
    #
    #     run_safe(arrhythmia, backends, delta)
    #     run_safe(astro, backends, delta)
    #     run_safe(dishwasher, backends, delta)
    #     run_safe(eeg, backends, delta)
    #     run_safe(gap, backends, delta)
    #     run_safe(pamap, backends, delta)
    #     run_safe(penguin, backends, delta, use_1m=True)
    #     run_safe(penguin, backends, delta, use_1m=False)


    backends = ["scalable"]  # , "scalable"
    subsamplings = [16, 8, 4, 2]
    for subsampling in subsamplings:
        print(f"Using subsampling {subsampling}")
        arrhythmia.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
        astro.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
        dishwasher.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
        eeg.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
        gap.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
        pamap.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
        penguin.run_motiflets_scale_n(backends=backends, subsampling=subsampling, use_1m=True)
        penguin.run_motiflets_scale_n(backends=backends, subsampling=subsampling, use_1m=False)

    # backends = ["scalable"]
    # penguin.run_motiflets_scale_n(backends=backends, use_1m=True)
    # penguin.run_motiflets_scale_n(backends=backends, use_1m=False)
    # astro.run_motiflets_scale_n(backends=backends)
    # arrhythmia.run_motiflets_scale_n(backends=backends)
    # dishwasher.run_motiflets_scale_n(backends=backends)
    # eeg.run_motiflets_scale_n(backends=backends)
    # gap.run_motiflets_scale_n(backends=backends)
    # pamap.run_motiflets_scale_n(backends=backends)

if __name__ == "__main__":
    main()
