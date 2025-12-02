import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/motifs'

import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import numpy as np
import utils as ut


deltas = [0.1]
k_max = 20

memory = {
    # key, filename, momp motif length, momp motif meaning, dataset length
    "EOG_one_hour_50_Hz": "8 GB",
    "Challenge2009TestSetA_101a": "8 GB",
    "swtAttack7": "8 GB",
    "swtAttack38": "8 GB",
    "BlackLeggedKittiwake": "8 GB",
    "stator_winding": "8 GB",
    "EOG_one_hour_400_Hz": "8 GB",
    "Challenge2009Respiration500HZ": "8 GB",
    "HAR_Ambient_Sensor_Data": "8 GB",
    "water": "8 GB",
    "SpainishEnergyDataset": "8 GB",
    "house": "8 GB",
    "WindTurbine": "8 GB",
    "MGHSleepElectromyography": "8 GB",
    "CinC_Challenge": "8 GB",
    "Lab_K_060314": "8 GB",
    "Lab_FD_061014": "8 GB",
    "solarwind": "8 GB",
    "Bird12-Week3_2018_1_10":"8 GB",
    # "FingerFlexionECoG": "8 GB",
    #"SpainishEnergyDataset5sec": "8 GB",
    #"lorenzAttractorsLONG": "8 GB",
    #"recorddata": "8 GB",
    #"SynchrophasorEventsLarge": "8 GB",
}

def file_exists(ds_name):
    directory = "results"

    file_exists = False
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and ds_name in filename:
            file_exists = True
            break

    if file_exists:
        print(f"A file with {ds_name} exists in the directory.")
    else:
        print(f"No matching file for {ds_name} found.")

    return file_exists

def main():
    lengths = np.array([properties[-1] for properties in list(ut.filenames.values())])
    sorted_idx = np.argsort(lengths)
    for filename in np.array(list(ut.filenames.keys()))[sorted_idx]:
        # if filename in [
        #     "swtAttack7",
        #     "stator_winding",
        #     "Challenge2009Respiration500HZ",
        #     "house",
        #     "WindTurbine",
        #     "MGHSleepElectromyography",
        #     "Lab_K_060314",
        #     "Lab_FD_061014",
        #     "solarwind",
        #     "Week3_2018_1_10",
        #     "Bird12-Week3_2018_1_10",
        #     "FingerFlexionECoG",
        #     "SpainishEnergyDataset5sec",
        #     "lorenzAttractorsLONG",
        #     "SynchrophasorEventsLarge"
        # ]:
        #     continue

        if not file_exists(filename):
            ds_name, length, _, _ = ut.filenames[filename]
            if not np.isnan(length):
                print(f"Running: {filename, ds_name}")
                data = ut.read_mat(filename)
                pd.DataFrame(data).to_csv(f"csv/{ds_name}_ts.csv", index=False,
                                        float_format='%.4f')

                # # pyattimo
                # backend = "pyattimo"
                # for delta in deltas:
                #     memory_value = memory[filename]
                #     ut.run_safe(
                #         ds_name=filename,
                #         series=data,
                #         l_range=[length],
                #         k_max=k_max,
                #         backend=backend,
                #         pyattimo_delta=delta,
                #         pyattimo_max_memory=memory_value
                #     )
            else:
                print(f"Skipping {filename} due to unknown length.")


if __name__ == "__main__":
    main()
