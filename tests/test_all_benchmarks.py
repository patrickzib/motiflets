import test_pyattimo_gap as gap
import test_pyattimo_pamap as pamap
import test_pyattimo_penguin as penguin
import test_pyattimo_astro as astro
import test_pyattimo_dishwasher as dishwasher
import test_pyattimo_eeg_physiodata as eeg
import test_pyattimo_arrhythmia as arrhythmia

def main():
    backends = ["pyattimo"]
    deltas = [0.9, 0.5, 0.1]

    for delta in deltas:
        print(f"Using delta {delta}")
        gap.test_motiflets_scale_n(backends=backends, delta=delta)
        pamap.test_motiflets_scale_n(backends=backends, delta=delta)
        penguin.test_motiflets_scale_n(backends=backends, delta=delta)
        astro.test_motiflets_scale_n(backends=backends, delta=delta)
        dishwasher.test_motiflets_scale_n(backends=backends, delta=delta)
        eeg.test_motiflets_scale_n(backends=backends, delta=delta)
        arrhythmia.test_motiflets_scale_n(backends=backends, delta=delta)

if __name__ == "__main__":
    main()
