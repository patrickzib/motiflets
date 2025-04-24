import traceback

import pandas as pd
import scipy.io as sio

import utils as ut

path = "../datasets/momp/"

filenames = {
    "Bird12": "Bird12-Week3_2018_1_10",
    "BlackLeggedKittiwake": "BlackLeggedKittiwake",
    "Challenge2009Respiration500HZ": "Challenge2009Respiration500HZ",
    "Challenge2009TestSetA_101a": "Challenge2009TestSetA_101a",
    "CinC_Challenge": "CinC_Challenge",
    "EOG_one_hour_50_Hz": "EOG_one_hour_50_Hz",
    "EOG_one_hour_400_Hz": "EOG_one_hour_400_Hz",
    "FingerFlexionECoG": "FingerFlexionECoG",
    "HAR_Ambient_Sensor_Data": "HAR_Ambient_Sensor_Data",
    "house": "house",
    "Lab_FD_061014": "Lab_FD_061014",
    "Lab_K_060314": "Lab_K_060314",
    "lorenzAttractorsLONG": "lorenzAttractorsLONG",
    "MGHSleepElectromyography": "MGHSleepElectromyography",
    "recorddata": "recorddata",
    "solarwind": "solarwind",
    "SpainishEnergyDataset": "SpainishEnergyDataset",
    "SpainishEnergyDataset5sec": "SpainishEnergyDataset5sec",
    "stator_winding": "stator_winding",
    "swtAttack7": "swtAttack7",
    "swtAttack38": "swtAttack38",
    "SynchrophasorEventsLarge": "SynchrophasorEventsLarge",
    "water": "water",
    "WindTurbine": "WindTurbine"
}


def read_mat(filename):
    print (f"Reading from {path + filename + '.mat'}")
    data = sio.loadmat(path + filename + '.mat')
    key = list(data.keys())[3]
    series = pd.DataFrame(data[key])
    return series.to_numpy().flatten()


def test_motiflets_scale_n(
        ds_name, data,
        n,
        l_range,
        k_max,
        backends=None,
        delta=None,
        subsampling=None,
):
    def pass_data():
        return ds_name, data

    ut.test_motiflets_scale_n(
        read_data=pass_data,
        n_range=[n],
        l_range=l_range,
        k_max=k_max,
        backends=backends,
        delta=delta,
        subsampling=subsampling
    )


def run_safe(ds_name, series, l_range, k_max, backends, delta, subsampling=None):
    try:
        # n = 10000  # len(series)
        test_motiflets_scale_n(ds_name, series, n, l_range, k_max, backends=backends, delta=delta, subsampling=subsampling)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


l_range = [128, 256, 512, 1024, 2048]
# ks = [5, 10, 20]
deltas = [None, 0.25]

def main():
    for ds_name in filenames.keys():
        filename = filenames[ds_name]

        backends = ["pyattimo"]
        for delta in deltas:
            run_safe(
                ds_name, read_mat(filename), l_range, 10, backends, delta
            )

        backends = ["scalable"]
        run_safe(
            ds_name, read_mat(filename), l_range, 10, backends, delta
        )

        backends = ["scalable"]
        run_safe(
            ds_name, read_mat(filename), l_range, 10, backends, delta
        )

        run_safe(
            ds_name, read_mat(filename), l_range, 10, backends, delta, subsampling=16
        )

        # break


if __name__ == "__main__":
    main()
