import os
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from plotting import Motiflets, plot_motifset
import utils as ut

interesting = {
    "HAR": "HAR_Ambient_Sensor_Data",
    "Household": "house",
    "Solarwind": "solarwind",
    "SpanishED": "SpainishEnergyDataset5sec",
    "Water Demand": "water",
    "Finger ECOG": "FingerFlexionECoG"
}

# MOMP
MOMP_results = {
    # key: [[first_index, second_index], motif_length]
    "EOG_one_hour_50_Hz": [[62478, 83143], 1024],
    "EOG_one_hour_400_Hz": [[236605, 1321975], 8192],
    "CinC_Challenge": [[4823203, 4828203], 8192],
    "swtAttack7": [[133860, 144969], 16384],
    "MGHSleepElectromyography": [[1964375, 5889161], 32768],
    "recorddata": [[3540784, 3998440], 2048],
    "Challenge2009Respiration500HZ": [[62963, 1296128], 16384],
    "Lab_K_060314": [[6817003, 6979308], 65536],
    "Bird12-Week3_2018_1_10": [[80420, 1374748], 16384],
    "BlackLeggedKittiwake": [[45696, 742777], 8192],
    "water": [[973390, 1255811], 8192],
    "FingerFlexionECoG": [[14395771, 19195771], 16384],
    "SpainishEnergyDataset5sec": [[12884629, 13368457], 524288],
    "lorenzAttractorsLONG": [[1, 15361793], 524288],
    "stator_winding": [[336089, 392549], 32768],
    "solarwind": [[2221814, 5071908], 32768],
    "WindTurbine": [[929205, 1226751], 32768],
    "SynchrophasorEventsLarge": [[30610002, 61714007], 65536],
    "Lab_FD_061014": [[6684313, 7342637], 32768],
    "house": [[617128, 928816], 32768],
    "HAR_Ambient_Sensor_Data": [[425966, 1615881], 4096],
    "Challenge2009TestSetA_101a": [[353601, 424237], 4096],
    "swtAttack38": [[311254, 319874], 4096]
}

def test_plot():
    points_to_plot = 10_000

    momp_path = f"results/pyattimo_momp/"
    for ds_name, _ in MOMP_results.items():
        if f"{ds_name}.json" in os.listdir(momp_path):
            ts = ut.read_mat(ds_name)

            json_path = f"{momp_path}{ds_name}.json"
            df = pd.read_json(json_path)

            motif_sets = df["motiflet"][0]
            motif_length = df["motif length"][0]
            elbows = df["elbows"][0]
            elbows.append(len(motif_sets)-1)
            motiflets = np.array(motif_sets, dtype=object)[elbows]

            fig, ax = plot_motifset(
                ds_name,
                ts,
                motifsets=list(map(np.array, motiflets)),
                max_points=points_to_plot,
                motif_length=motif_length,
                show=False,
            )

            path = f"images/datasets/MOMP_pyattimo_{ds_name.lower().replace(' ', '_')}.pdf"
            plt.savefig(path)
            plt.show()


    for name, ds_name in interesting.items():
        filename = interesting[name]
        ts = ut.read_mat(filename)

        ml = Motiflets(ds_name, ts)

        # ml.plot_dataset(
        #     max_points=points_to_plot,
        #     path=f"images/datasets/{name.lower().replace(' ', '_')}.pdf"
        # )

        momp = MOMP_results[filename]
        motiflets = np.array([momp[0]])
        dists = [0]
        motif_length = momp[1]

        fig, ax = plot_motifset(
            ds_name,
            ts,
            motifsets=motiflets,
            max_points=points_to_plot,
            motif_length=motif_length,
            show=False,
        )

        path = f"images/datasets/MOMP_{name.lower().replace(' ', '_')}.pdf"
        plt.savefig(path)
        plt.show()


