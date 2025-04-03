import traceback

import test_pyattimo_gap as gap
import test_pyattimo_pamap as pamap
import test_pyattimo_penguin as penguin
import test_pyattimo_astro as astro
import test_pyattimo_dishwasher as dishwasher
import test_pyattimo_eeg_physiodata as eeg
import test_pyattimo_arrhythmia as arrhythmia


def main():
    backends = ["pyattimo"]
    deltas = [0.10]

    for delta in deltas:
        print(f"Using delta {delta}")

        try:
           astro.test_motiflets_scale_n(backends=backends, delta=delta)
        except Exception as e:
           print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
            arrhythmia.test_motiflets_scale_n(backends=backends, delta=delta)
        except Exception as e:
            print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
           dishwasher.test_motiflets_scale_n(backends=backends, delta=delta)
        except Exception as e:
           print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
           eeg.test_motiflets_scale_n(backends=backends, delta=delta)
        except Exception as e:
           print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
           gap.test_motiflets_scale_n(backends=backends, delta=delta)
        except Exception as e:
           print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
           pamap.test_motiflets_scale_n(backends=backends, delta=delta)
        except Exception as e:
           print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
           penguin.test_motiflets_scale_n(backends=backends, delta=delta, use_1m=True)
        except Exception as e:
           print(traceback.format_exc())
        except BaseException as e:
           print(f"Caught a panic: {e}")

        try:
            penguin.test_motiflets_scale_n(backends=backends, delta=delta, use_1m=False)
        except Exception as e:
            print(traceback.format_exc())
        except BaseException as e:
            print(f"Caught a panic: {e}")

if __name__ == "__main__":
    main()
