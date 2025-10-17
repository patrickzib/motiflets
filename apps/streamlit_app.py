"""Interactive Streamlit front-end for Motiflets."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from motiflets.motiflets import search_k_motiflets_elbow

st.set_page_config(page_title="Motiflets Explorer", layout="wide")

def _seed_from_params(*values: object) -> int:
    """Return a deterministic seed based on the provided values."""
    return abs(hash(values)) % (2**63)


@st.cache_data(show_spinner=False)
def generate_sine_wave(length: int, noise: float) -> np.ndarray:
    """Create a noisy sine wave with reproducible randomness."""
    rng = np.random.default_rng(_seed_from_params(length, round(noise, 3), "sine"))
    x = np.linspace(0, 8 * np.pi, length, dtype=np.float64)
    series = np.sin(x)
    if noise > 0:
        series = series + rng.normal(scale=noise, size=length)
    return series.astype(np.float64, copy=False)


@st.cache_data(show_spinner=False)
def generate_random_walk(length: int, noise: float) -> np.ndarray:
    """Create a random walk that is z-normalised."""
    rng = np.random.default_rng(_seed_from_params(length, round(noise, 3), "walk"))
    steps = rng.normal(loc=0.0, scale=noise if noise > 0 else 0.1, size=length)
    series = steps.cumsum()
    series = (series - series.mean()) / max(series.std(ddof=0), 1e-6)
    return series.astype(np.float64, copy=False)


def _as_1d(series: np.ndarray) -> np.ndarray:
    """Flatten higher dimensional inputs to one dimension."""
    array = np.asarray(series, dtype=np.float64)
    if array.ndim == 1:
        return array
    if 1 in array.shape:
        return np.ravel(array)
    # Fallback to first dimension for multivariate uploads.
    return array.reshape(array.shape[0], -1)[0]


def load_uploaded_series(uploaded_file) -> Optional[np.ndarray]:
    """Parse user uploaded files into a 1D numpy array."""
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix in {".csv", ".txt", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else None
            df = pd.read_csv(uploaded_file, sep=sep)
            numeric = df.select_dtypes(include=["number"]).squeeze("columns")
            return _as_1d(numeric.to_numpy())
        if suffix == ".npy":
            data = np.load(uploaded_file, allow_pickle=False)
            return _as_1d(data)
        if suffix == ".npz":
            archive = np.load(uploaded_file, allow_pickle=False)
            first_key = list(archive.keys())[0]
            return _as_1d(archive[first_key])
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to parse uploaded data: {exc}")
        return None

    st.error("Unsupported file type. Please upload CSV, TSV, NPY, or NPZ files.")
    return None


def z_normalise(series: np.ndarray) -> np.ndarray:
    """Z-normalise a copy of the provided series."""
    arr = np.asarray(series, dtype=np.float64)
    std = float(arr.std(ddof=0))
    if std < 1e-12:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def prepare_dataset(selection: str, parameters: Dict[str, float]) -> Tuple[np.ndarray, str]:
    """Return the selected dataset and label."""
    if selection == "Synthetic · Sine wave":
        length = int(parameters["length"])
        noise = float(parameters["noise"])
        series = generate_sine_wave(length, noise)
        label = f"Sine (n={length}, noise={noise:.2f})"
        return series, label

    if selection == "Synthetic · Random walk":
        length = int(parameters["length"])
        noise = float(parameters["noise"])
        series = generate_random_walk(length, noise)
        label = f"Random walk (n={length}, noise={noise:.2f})"
        return series, label

    if selection == "Upload your own":
        uploaded = parameters.get("uploaded")
        series = load_uploaded_series(uploaded)
        label = uploaded.name if uploaded else "Uploaded dataset"
        return series, label

    raise ValueError(f"Unknown dataset selection: {selection}")


def format_positions(candidate: Optional[np.ndarray]) -> List[int]:
    if candidate is None:
        return []
    positions = np.asarray(candidate, dtype=np.int64)
    positions = positions[positions >= 0]
    return positions.tolist()


def plot_motif_segments(series: np.ndarray, motif_length: int, positions: List[int], title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, color="tab:blue", linewidth=1.0)
    for pos in positions:
        ax.axvspan(pos, pos + motif_length, color="tab:orange", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    st.pyplot(fig, clear_figure=True)


def run_motiflets(
        series: np.ndarray,
        k_max: int,
        motif_mode: str,
        motif_length: int,
        motif_range: Optional[np.ndarray],
        backend: str,
        slack: float,
        n_jobs: int,
        elbow_deviation: float,
        filter_overlaps: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    kwargs = dict(
        k_max=k_max,
        data=series,
        elbow_deviation=elbow_deviation,
        filter=filter_overlaps,
        slack=slack,
        n_jobs=n_jobs,
        backend=backend,
    )
    if motif_mode == "Auto (AU_EF)":
        kwargs["motif_length"] = "auto"
        kwargs["motif_length_range"] = motif_range
    else:
        kwargs["motif_length"] = int(motif_length)
    return search_k_motiflets_elbow(**kwargs)


SESSION_RESULTS_KEY = "motiflets_results"


def main() -> None:
    st.title("Motiflets Explorer")
    st.caption("Interactively discover motifs using the k-Motiflets algorithm.")

    with st.sidebar:
        st.header("Dataset")
        dataset_choice = st.selectbox(
            "Choose data source",
            [
                "Synthetic · Sine wave",
                "Synthetic · Random walk",
                "Upload your own",
            ],
        )

        dataset_params: Dict[str, object] = {}
        if dataset_choice.startswith("Synthetic"):
            dataset_params["length"] = st.slider(
                "Series length",
                min_value=256,
                max_value=8192,
                value=2048,
                step=128,
            )
            dataset_params["noise"] = st.slider(
                "Noise level",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
            )
        else:
            dataset_params["uploaded"] = st.file_uploader(
                "Upload a CSV, TSV, NPY or NPZ file",
                type=["csv", "txt", "tsv", "npy", "npz"],
            )

        st.header("Preprocessing")
        normalise = st.checkbox("Z-normalise", value=True)
        limit_points = st.slider(
            "Use first n points",
            min_value=128,
            max_value=16384,
            value=4096,
            step=128,
            help="Limit the series length to keep experiments responsive.",
        )

        st.header("Motif parameters")
        k_max = st.slider("k max", min_value=3, max_value=20, value=6, step=1)
        backend = st.selectbox(
            "Backend",
            options=["scalable", "default", "pyattimo", "sparse"],
            help="Advanced vector-search backends (faiss, annoy, etc.) can be added later.",
        )
        n_jobs = st.slider(
            "Number of jobs",
            min_value=1,
            max_value=min(8, os.cpu_count() or 1),
            value=1,
            step=1,
        )
        slack = st.slider(
            "Slack (exclusion zone)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        elbow_deviation = st.slider(
            "Elbow deviation",
            min_value=1.0,
            max_value=1.5,
            value=1.05,
            step=0.01,
        )
        filter_overlaps = st.checkbox(
            "Filter overlapping motif sets",
            value=True,
        )

        motif_mode = st.radio(
            "Motif length",
            ["Manual", "Auto (AU_EF)"],
            horizontal=True,
        )
        motif_length = 128
        motif_range = None
        if motif_mode == "Manual":
            motif_length = st.slider(
                "Motif length (m)",
                min_value=32,
                max_value=1024,
                value=128,
                step=16,
            )
        else:
            range_start, range_end = st.slider(
                "Motif length search range",
                min_value=32,
                max_value=2048,
                value=(64, 256),
                step=16,
            )
            range_step = st.number_input(
                "Range step",
                min_value=8,
                max_value=256,
                value=16,
                step=8,
            )
            motif_range = np.arange(range_start, range_end + range_step, range_step, dtype=np.int32)

        run_button = st.button("Run motif discovery", type="primary")

    try:
        data, dataset_label = prepare_dataset(dataset_choice, dataset_params)
    except ValueError as exc:
        st.error(str(exc))
        return

    if data is None or len(data) == 0:
        st.info("Provide a dataset to begin.")
        return

    series = _as_1d(data)
    series = series[: min(len(series), int(limit_points))]
    if normalise:
        series = z_normalise(series)

    st.subheader("Dataset overview")
    st.write(
        {
            "Name": dataset_label,
            "Length": len(series),
            "Min": float(series.min()),
            "Max": float(series.max()),
        }
    )
    st.line_chart(pd.DataFrame({"value": series}))

    if run_button:
        if motif_mode == "Auto (AU_EF)" and (motif_range is None or motif_range.size == 0):
            st.error("The motif length range is empty. Adjust the slider or step size.")
        elif len(series) <= 2 * (motif_range[0] if motif_range is not None else motif_length):
            st.error("Increase the series length or reduce the motif length.")
        else:
            with st.spinner("Searching for motiflets..."):
                results = run_motiflets(
                    series=series,
                    k_max=k_max,
                    motif_mode=motif_mode,
                    motif_length=motif_length,
                    motif_range=motif_range,
                    backend=backend,
                    slack=slack,
                    n_jobs=n_jobs,
                    elbow_deviation=elbow_deviation,
                    filter_overlaps=filter_overlaps,
                )
            st.session_state[SESSION_RESULTS_KEY] = {
                "distances": results[0],
                "candidates": results[1],
                "elbows": results[2],
                "motif_length": int(results[3]),
                "memory_usage": float(results[4]),
                "series": series,
                "dataset_label": dataset_label,
                "params": {
                    "k_max": k_max,
                    "backend": backend,
                    "slack": slack,
                    "n_jobs": n_jobs,
                    "elbow_deviation": elbow_deviation,
                    "motif_mode": motif_mode,
                },
            }

    results = st.session_state.get(SESSION_RESULTS_KEY)
    if not results:
        return

    st.subheader("Motif discovery results")
    st.write(
        f"Motif length used: **{results['motif_length']}** | Memory: {results['memory_usage']:.2f} MB"
    )

    distances = np.asarray(results["distances"], dtype=np.float64)
    candidates = np.asarray(results["candidates"], dtype=object)
    elbows = set(int(k) for k in np.asarray(results["elbows"], dtype=np.int64))

    ks = np.arange(len(distances), dtype=int)
    table_rows = []
    for k in ks:
        if k < 2:
            continue
        extent = float(distances[k]) if np.isfinite(distances[k]) else None
        positions = format_positions(candidates[k] if k < len(candidates) else None)
        table_rows.append({
            "k": int(k),
            "extent": extent,
            "is_elbow": k in elbows,
            "motif_positions": positions,
        })

    if not table_rows:
        st.warning("No motiflets were found for the current configuration.")
        return

    df = pd.DataFrame(table_rows)
    st.dataframe(df, width='stretch')

    chart_df = df.set_index("k")["extent"].dropna()
    if not chart_df.empty:
        st.line_chart(chart_df)

    options = df["k"].tolist()
    default_index = 0
    elbow_candidates = [k for k in options if k in elbows]
    if elbow_candidates:
        default_index = options.index(elbow_candidates[0])
    selected_k = st.selectbox("Select k to visualise", options=options, index=default_index)

    selected_positions = format_positions(candidates[selected_k])
    if selected_positions:
        plot_motif_segments(
            series=results["series"],
            motif_length=results["motif_length"],
            positions=selected_positions,
            title=f"Top-{selected_k} motif instances",
        )
    else:
        st.info("Selected motif set has no valid positions to visualise.")

    export_payload = {
        "dataset": results["dataset_label"],
        "motif_length": results["motif_length"],
        "k_results": table_rows,
        "elbow_points": sorted(elbows),
        "parameters": results["params"],
    }

    st.download_button(
        label="Download results as JSON",
        data=json.dumps(export_payload, indent=2),
        file_name="motiflets_results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
