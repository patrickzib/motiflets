import traceback

import test_pyattimo_gap as gap
import test_pyattimo_pamap as pamap
import test_pyattimo_penguin as penguin
import test_pyattimo_astro as astro
import test_pyattimo_dishwasher as dishwasher
import test_pyattimo_eeg_physiodata as eeg
import test_pyattimo_arrhythmia as arrhythmia


def run_safe(module, backends, delta, use_1m=None):
    try:
        if use_1m:
            module.test_motiflets_scale_n(backends=backends, delta=delta, use_1m=use_1m)
        else:
            module.test_motiflets_scale_n(backends=backends, delta=delta)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


def main():
    backends = ["pyattimo"]
    deltas = [0.10, 0.25, 0.50]

    for delta in deltas:
        print(f"Using delta {delta}")

        run_safe(astro, backends, delta)
        run_safe(arrhythmia, backends, delta)
        run_safe(dishwasher, backends, delta)
        run_safe(eeg, backends, delta)
        run_safe(gap, backends, delta)
        run_safe(pamap, backends, delta)
        run_safe(penguin, backends, delta, use_1m=True)
        run_safe(penguin, backends, delta, use_1m=False)


    backends = ["default"]
    subsamplings = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for subsampling in subsamplings:
        print(f"Using subsampling {subsampling}")
        astro.test_motiflets_scale_n(backends=backends, subsampling=subsampling)
        arrhythmia.test_motiflets_scale_n(backends=backends, subsampling=subsampling)
        dishwasher.test_motiflets_scale_n(backends=backends, subsampling=subsampling)
        eeg.test_motiflets_scale_n(backends=backends, subsampling=subsampling)
        gap.test_motiflets_scale_n(backends=backends, subsampling=subsampling)
        pamap.test_motiflets_scale_n(backends=backends, subsampling=subsampling)
        penguin.test_motiflets_scale_n(backends=backends, subsampling=subsampling, use_1m=True)
        penguin.test_motiflets_scale_n(backends=backends, subsampling=subsampling, use_1m=False)

    backends = ["scalable, default"]
    astro.test_motiflets_scale_n(backends=backends)
    arrhythmia.test_motiflets_scale_n(backends=backends)
    dishwasher.test_motiflets_scale_n(backends=backends)
    eeg.test_motiflets_scale_n(backends=backends)
    gap.test_motiflets_scale_n(backends=backends)
    pamap.test_motiflets_scale_n(backends=backends)
    penguin.test_motiflets_scale_n(backends=backends, use_1m=True)
    penguin.test_motiflets_scale_n(backends=backends, use_1m=False)


if __name__ == "__main__":
    main()
