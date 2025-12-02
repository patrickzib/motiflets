import glob
import os
import numpy as np

from motiflets.plotting import Motiflets

import pandas as pd

import run_gap as gap
import run_pamap as pamap
import run_penguin as penguin
import run_astro as astro
import run_dishwasher as dishwasher
import run_eeg_physiodata as eeg
import run_arrhythmia as arrhythmia


def get_results():
    directory = "results/other_benchmarks"
    json_files = glob.glob(os.path.join(directory, "*.json"))
    return json_files


def plot_motifset(
        dataset_name,
        read_data,
        motifsets,
        motif_length,
        elbow_points,
        delta,
        k
    ):
    ds_name, series = read_data()
    ml = Motiflets(ds_name, series)
    ml.motif_length = motif_length
    ml.motiflets = motifsets

    file_name = f"results/images/others/{dataset_name}_{motif_length}_d_{delta}_k_{k}.pdf"
    if not os.path.isfile(file_name):
        points_to_plot = 10_000
        ml.plot_motifset(
            max_points=points_to_plot,
            elbow_point=elbow_points,
            path=file_name)


def main():
    json_files = get_results()

    for json_file in json_files:
        print(f"Processing file: {json_file}")

        results = pd.read_json(json_file)

        start = json_file.find("scalability_n_")
        end = json_file.find("_pyattimo")
        dataset_name = json_file[start + len("scalability_n_"):end-3]

        start_delta = json_file.find("delta_")
        end_delta =  json_file.find(".json")
        delta = json_file[start_delta + len("delta_"):end_delta]

        start_k = json_file.find(dataset_name+"_")
        end_k = json_file.find("_pyattimo_delta")
        k = json_file[start_k + len(dataset_name+"_"):end_k]

        print(f"Dataset name extracted: {dataset_name} {delta} {k}")

        if dataset_name == "ASTRO":
            read_data = astro.read_data
        elif dataset_name == "Dishwasher":
            read_data = dishwasher.read_data
        elif dataset_name == "PAMAP":
            read_data = pamap.read_data
        elif dataset_name == "GAP":
            read_data = gap.read_data
        elif dataset_name == "EEG-Sleep":
            read_data = eeg.read_data
        elif dataset_name == "Penguin3M":
            read_data = penguin.read_penguin_3m
        elif dataset_name == "Penguin1M":
            read_data = penguin.read_penguin_1m_channel0
        elif dataset_name == "Arrhythmia":
            read_data = arrhythmia.read_data
        else:
            print(f"Unknown dataset name: {dataset_name}")
            continue

        for index, row in results.iterrows():
            print(f"Processing row: {index}")
            motif_length = row["motif length"]
            motifsets = np.array([np.array(motifset, dtype=object) for motifset in row["motiflet"]], dtype=object)
            elbow_points = np.array(row["elbows"])

            plot_motifset(
                dataset_name,
                read_data,
                motifsets,
                motif_length,
                elbow_points,
                delta,
                k
            )


if __name__ == "__main__":
    main()

