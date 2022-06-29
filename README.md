# Motiflets

This page was built in support of our paper "Motiflets - Simple and Accurate Detection of Motifs in Time Series" by Patrick Schäfer and Ulf Leser.

The paper is completely self-contained, the purpose of this webpage is to provide the 
k-Motiflet code, and the raw data to readers of the paper.

Supporting Material
- `notebooks`: Please see the Jupyter Notebooks for use cases
- `csvs`: The results of the scalability experiments
- `motiflets`: Code implementing k-Motiflet
- `datasets`: Use cases in the paper
- `jars`: Java code of the competitors used in out paper: EMMA, Latent Motifs and Set Finder.

# k-Motiflets

Intuitively speaking, k-Motiflets are the largest set of exactly k similar subsequences.

# Showcase

The following video highlights the ease of use of $k$-Motiflets using an ECG recording from the Long Term Atrial Fibrillation (LTAF) database.

**In essence, there is no need for tuning any real-valued similarity threshold via trial-and-error, as is teh case for virtually all motif set competitors. 
Instead, for $k$-Motiflets we may either directly set the maximal number of repetitions $k$ of a motif, or simply learn this value from the data.**

https://user-images.githubusercontent.com/7783034/173186103-c8b6302e-2434-4a09-89f4-ddad2e63f997.mp4

# Usage

Here we illustrate how to use k-Motiflets. 

The following TS is an ECG from the Long Term Atrial Fibrillation (LTAF) database, which 
is often used for demonstrations in motif discovery (MD). The problem is particularly 
difficult for MD as actually two motifs exists: The first half of the TS contains a 
rectangular calibration signal with 6 occurrences, and the second half shows ECG 
heartbeats with 16 to 17 occurrences. 

![The ECG heartbeat dataset](images/ts_ecg.png)

The major challenges in motif discovery are to learn the length of interesting motifs
and to find the largest set of the same motif, i.e. all repetitions.

# Learning the motif length `l`

We first extract meaningful **motif lengths (l)** from this use case:

```
ks = 20
length_range = np.arange(25,200,25) 
motif_length = plot_motif_length_selection(
    ks, series, file, 
    motif_length_range=length_range, ds_name=ds_name)
```
<img src="images/plot_au_ef.png" width="300">

The plot shows that meaningful motifs are within a range of 0.8s to 1s, equal
to roughly a heartbeat rate of 60-80 bpm.

# Learning the motif size `k`

To extract meaningful **motif sizes (k)** from this use case, we run 

```
dists, motiflets, elbow_points = plot_elbow(
    ks, series, file, ds_name=ds_name, plot_elbows=True,
    motif_length=motif_length, method_name="K-Motiflets", ground_truth=df_gt)
```

The variable `elbow_points` holds characteristic motif sizes found.  
Elbow points represent meaningful motif sizes. Here, $6$ and $16$ are elbows, which are 
the 6 calibration waves and the 16 heartbeats.

<img src="images/elbows.png" width="300">

We finally plot these motifs:

<img src="images/motiflets.png" width="600">

The first repetitions perfectly match the calibration signal (orange), while the latter 16 
repetitions perfectly match the ECG waves (green).

# Use Cases

Data Sets: We collected five challenging real-life data sets to assess the quality and scalability of MD algorithms. For three out of these, the literature describes the existence of motifs though without actually annotating them. An overview can be found in Table 2. The five data sets are the following:

- Jupyter-Notebook <a href="https://github.com/patrickzib/motiflets/blob/435eec7f1f70ae81b8ec246ffc38842fce869660/notebooks/use_cases_paper.ipynb">Use Cases for k-Motiflets</a>: highlights all use cases used in the paper and shows the unique ability of k-Motiflets to learn its parameters from the data and find itneresting motif sets.
- Jupyter-Notebook <a href="https://github.com/patrickzib/motiflets/blob/2303f246edd3383bb7836b5e5b051ec201825633/notebooks/use_cases_motif_sets_muscle_activation.ipynb">Muscle Activation</a> was collected from professional in-line speed skating on a large motor driven treadmill with Electromyo- graphy (EMG) data of multiple movements. It consists of 29.899 measurements at 100Hz corresponding to 30s in total. The known motifs are the muscle movement and a recovery phase.
- Jupyter-Notebook <a href="https://github.com/patrickzib/motiflets/blob/2303f246edd3383bb7836b5e5b051ec201825633/notebooks/use_cases_motif_sets_ecg.ipynb">ECG Heartbeats</a> contains a patient’s (with ID 71) heartbeat from the LTAF database. It consists of 3.000 measurements at 128𝐻𝑧 corresponding to 23𝑠. The heartbeat rate is around 60 to 80 bpm. There are two motifs: A calibration signal and the actual heartbeats.
- Jupyter-Notebook <a href="https://github.com/patrickzib/motiflets/blob/2303f246edd3383bb7836b5e5b051ec201825633/notebooks/use_cases_motif_sets_physiodata-spindles.ipynb">Physiodata - EEG sleep data</a> contains a recording of an after- noon nap of a healthy, nonsmoking person, between 20 to 40 years old [10]. Data was recorded with an extrathoracic strain belt. The dataset consists of 269.286 points at 100H𝑧 corresponding to 45𝑚𝑖𝑛. Known motifs are so-called sleep spindles and 𝑘-complexes.
- Jupyter-Notebook <a href="https://github.com/patrickzib/motiflets/blob/2303f246edd3383bb7836b5e5b051ec201825633/notebooks/use_cases_motif_sets_winding.ipynb">Industrial Winding Process</a> is a snapshot of a process where a plastic web is unwound from a first reel (unwinding reel), goes over the second traction reel and is finally rewound on the the third rewinding reel. The recordings correspond to the traction of the second reel angular speed. The data contains 2.500 points sampled at 0.1𝑠, corresponding to 250𝑠. No documented motifs exist.
- Jupyter-Notebook <a href="https://github.com/patrickzib/motiflets/blob/2303f246edd3383bb7836b5e5b051ec201825633/notebooks/use_cases_fnirs.ipynb">Functional near-infrared spectroscopy (fNIRS)</a> contains brain imag- inary data recorded at 690𝑛𝑚 intensity. There are 208.028 measurements in total. The data is known to be a difficult example, as it contains four motion artifacts, due to movements of the patient, which dominate MD. No documented motifs exist.
