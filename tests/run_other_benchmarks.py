import traceback

import run_gap as gap
import run_pamap as pamap
import run_penguin as penguin
import run_astro as astro
import run_dishwasher as dishwasher
import run_eeg_physiodata as eeg
import run_arrhythmia as arrhythmia


def run_safe(module, backends, delta, k_max, use_1m=None):
    try:
        if use_1m is not None:
            module.run_motiflets_scale_n(backends=backends, delta=delta, use_1m=use_1m, k_max=k_max)
        else:
            module.run_motiflets_scale_n(backends=backends, delta=delta, k_max=k_max)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


def main():
    backends = ["scampi"]
    deltas = [0.1]
    k_maxs = [10, 20, 30, 40]

    for delta in deltas:
        for k_max in k_maxs:
            print(f"Using delta {delta}")

            #run_safe(penguin, backends, delta, k_max, use_1m=False)
            #run_safe(astro, backends, delta, k_max)
            #run_safe(arrhythmia, backends, delta, k_max)
            #run_safe(dishwasher, backends, delta, k_max)
            #run_safe(eeg, backends, delta, k_max)
            #run_safe(gap, backends, delta, k_max)
            #run_safe(pamap, backends, delta, k_max)
            run_safe(penguin, backends, delta, k_max, use_1m=True)


    # backends = ["scalable"]  # , "scalable"
    # subsamplings = [16, 8, 4, 2]
    # for subsampling in subsamplings:
    #     print(f"Using subsampling {subsampling}")
    #     arrhythmia.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
    #     astro.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
    #     dishwasher.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
    #     eeg.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
    #     gap.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
    #     pamap.run_motiflets_scale_n(backends=backends, subsampling=subsampling)
    #     penguin.run_motiflets_scale_n(backends=backends, subsampling=subsampling, use_1m=True)
    #     penguin.run_motiflets_scale_n(backends=backends, subsampling=subsampling, use_1m=False)

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
