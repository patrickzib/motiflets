import traceback

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats

import utils as ut

path = "../datasets/momp/"

filenames = {
    "Bird12-Week3_2018_1_10": ["22.5 hours of Chicken data at 100 Hz", 16384, ""],
    "BlackLeggedKittiwake": ["Flying Bird: Black‐legged Kittiwake", 8192, "?"],
    # "Challenge2009Respiration500HZ": ["Challenge 2009 Respiration", 16384, "?"],
    # "Challenge2009TestSetA_101a": ["Respiration", 4096, "?"],
    # "CinC_Challenge": ["Electroencephalography C3-M2 Part 2", 8192, "Calibration"],
    # "EOG_one_hour_50_Hz": ["EOG_one_hour_50_Hz", 2048, "?"],
    # "EOG_one_hour_400_Hz": ["EOG_one_hour_400_Hz", 8192, "?"],
    # "FingerFlexionECoG": ["Finger Flexion ECoG electrocorticography", 16384, "?"],
    # "HAR_Ambient_Sensor_Data": ["Human Activity Recognition", 4096, "?"],
    # "house": ["Household Electrical Demand", 32768, "?"],
    # "Lab_FD_061014": ["Insect EPG - Flaming Dragon", 32768, "?"],
    # "Lab_K_060314": ["ACP on Kryder Citrus", 65536, ""],
    # "lorenzAttractorsLONG": ["Lorenz Attractors", 524288, "?"],
    # "MGHSleepElectromyography": ["MGH Sleep Electromyography 200 Hz", 32768, "?"],
    # "recorddata": ["EOG Example", 2048, "?"],
    # "solarwind": ["Solar Wind", 32768, "?"],
    # "SpainishEnergyDataset": ["SpainishEnergyDataset", np.nan, "?"],
    # "SpainishEnergyDataset5sec": ["Spainish Energy Dataset 5 sec", 524288, "?"],
    # "stator_winding": ["Electric Motor Temperature", 32768, "?"],
    # "swtAttack7": ["Swat Attack 7", 4 * 16 * 256, "?"],
    # "swtAttack38": ["Swat Attack 38", 16 * 256, "?"],
    # "SynchrophasorEventsLarge": ["Synchrophasor Events Large", 256 * 2 * 128, "?"],
    # "water": ["Water Demand", 8192, ""],
    # "WindTurbine": ["Wind Turbine R24VMON Rotating system", 32768, "Precursor Dropout"]
}


def read_mat(filename):
    print (f"\tReading {filename} from {path + filename + '.mat'}")
    data = sio.loadmat(path + filename + '.mat')

    # extract data array
    key = list(data.keys())[3]

    # flatten output
    data = pd.DataFrame(data[key]).to_numpy().flatten()

    # data = scipy.stats.zscore(data)

    mb = (data.size * data.itemsize) / (1024 ** 2)

    print(f"\tLength: {len(data)} {mb:0.2f} MB")
    print(f"\tContains NaN or Inf? {np.isnan(data).any()} {np.isinf(data).any()}")
    print(f"\tStats Mean {np.mean(data):0.3f}, Std {np.std(data):0.3f} " +
          f"Min {np.min(data):0.3f} Max {np.max(data):0.3f}")

    # remove NaNs
    data = data[~np.isnan(data)]

    # remove Infs
    data = data[np.isfinite(data)]

    # np.savetxt(path + "/csv/" + filename + ".csv", data[:100000], delimiter=",")
    return data


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


def run_safe(ds_name, series, l_range, k_max, backends, delta=None, subsampling=None):
    try:
        n = 10_000  # len(series)
        test_motiflets_scale_n(
            ds_name, series, n,
            l_range=l_range, k_max=k_max, backends=backends, delta=delta, subsampling=subsampling)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


# 512 to 8192
l_range = [2**9, 2**10, 2**11]   # , 2**12, 2**13

# ks = [5, 10, 20]
deltas = [0.5]

k_max = 10

def main():

    for filename in filenames.keys():
        ds_name, length, meaning = filenames[filename]
        print (f"Running: {filename, ds_name}")

        # pyattimo
        # backends = ["pyattimo"]
        # for delta in deltas:
        #    run_safe(
        #        filename, read_mat(filename), l_range, k_max, backends, delta
        #    )

        # scalable
        backends = ["scalable"]
        run_safe(
           filename, read_mat(filename), l_range, k_max, backends
        )

        # # subsampling
        # backends = ["scalable"]
        # for subsampling in [8, 16]:
        #    run_safe(
        #       filename, read_mat(filename),
        #       l_range=l_range, k_max=k_max, backends=backends, subsampling=subsampling
        #    )


if __name__ == "__main__":
    main()
