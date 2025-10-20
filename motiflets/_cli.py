"""Command-line interface for motiflets."""

from __future__ import annotations

import argparse
import os
import sys
import io
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
from urllib.parse import urlparse
import urllib.request
from urllib.error import HTTPError

import pandas as pd

import motiflets.motiflets as ml

os.environ.setdefault("MPLBACKEND", "Agg")

_MOTIFLETS_CLASS = None


class CLIError(Exception):
    """Raised when the CLI encounters an unrecoverable user error."""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _dispatch(args)
    except CLIError as exc:
        print(f"motiflets: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        return 130


def _dispatch(args: argparse.Namespace) -> int:
    Motiflets = _get_motiflets_class()
    series = _load_series(
        args.source,
        column=_parse_column_identifier(args.column),
        index_column=_parse_column_identifier(args.index_column),
        dropna=not args.keep_na,
    )

    ds_name = args.ds_name or _infer_dataset_name(args.source)
    motiflets = Motiflets(
        ds_name=ds_name,
        series=series,
        elbow_deviation=args.elbow_deviation,
        distance=args.distance,
        slack=args.slack,
        n_jobs=args.n_jobs,
        backend=args.backend,
    )

    if args.command == "fit_motif_length":
        return _run_fit_motif_length(args, motiflets, series)
    if args.command == "fit_k_elbow":
        return _run_fit_k_elbow(args, motiflets, series)

    raise CLIError(f"unsupported command '{args.command}'")


def _get_motiflets_class():
    global _MOTIFLETS_CLASS
    if _MOTIFLETS_CLASS is None:
        from motiflets.plotting import Motiflets as motiflets_cls

        _MOTIFLETS_CLASS = motiflets_cls
    return _MOTIFLETS_CLASS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiflets",
        description="Run motiflets analyses from the command line.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_discover_version(),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "source",
        help="Path or URL to a CSV file containing the time series.",
    )
    common.add_argument(
        "--ds-name",
        help="Dataset name shown in summaries (defaults to filename stem).",
    )
    common.add_argument(
        "--column",
        help="Column to use as the time series (name or zero-based index).",
    )
    common.add_argument(
        "--index-column",
        help="Optional column to use as the time index (name or index).",
    )
    common.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep missing values instead of dropping them.",
    )
    common.add_argument(
        "--elbow-deviation",
        type=float,
        default=1.0,
        help="Minimal deviation to consider an elbow (default: 1.0).",
    )
    common.add_argument(
        "--slack",
        type=float,
        default=0.5,
        help="Slack percentage to avoid trivial matches (default: 0.5).",
    )
    common.add_argument(
        "--distance",
        default="znormed_ed",
        help="Distance metric name (default: znormed_ed).",
    )
    common.add_argument(
        "--backend",
        default="scalable",
        help="Backend to run motiflets with (default: scalable).",
    )
    common.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of jobs to use (-1 for auto).",
    )

    fit_motif_length = subparsers.add_parser(
        "fit_motif_length",
        parents=[common],
        help="Compute the best motif length for a dataset.",
    )
    fit_motif_length.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Search k in [2, k_max] (default: 10).",
    )
    fit_motif_length.add_argument(
        "--motif-length-range",
        nargs="+",
        help="Explicit motif length candidates (e.g. 48 96 144 or 24:192:24).",
    )
    fit_motif_length.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsampling factor to speed up computations (default: 2).",
    )

    fit_k_elbow = subparsers.add_parser(
        "fit_k_elbow",
        parents=[common],
        help="Search motif sets across different k.",
    )
    fit_k_elbow.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Search k in [2, k_max] (default: 10).",
    )
    fit_k_elbow.add_argument(
        "--motif-length",
        type=int,
        help="Use a fixed motif length (otherwise determined automatically).",
    )
    fit_k_elbow.add_argument(
        "--motif-length-range",
        nargs="+",
        help="Candidates to determine motif length when not provided.",
    )
    fit_k_elbow.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsampling factor to speed up automatic length search (default: 2).",
    )
    fit_k_elbow.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable filtering of overlapping motiflets.",
    )
    fit_k_elbow.add_argument(
        "--show-elbows",
        action="store_true",
        help="Display elbow points plot (disabled by default).",
    )
    fit_k_elbow.add_argument(
        "--show-grid",
        action="store_true",
        help="Display motif grid plots (disabled by default).",
    )

    return parser


def _run_fit_motif_length(
    args: argparse.Namespace,
    motiflets: "Motiflets",
    series: pd.Series,
) -> int:

    motif_length_range = _resolve_motif_length_range(
        args.motif_length_range,
        len(series),
    )

    best = motiflets.fit_motif_length(
        args.k_max,
        motif_length_range,
        subsample=args.subsample,
        plot=False
    )

    print(f"dataset:      {motiflets.ds_name}")
    print(f"observations: {len(series)}")
    print("motif_length_candidates: " + ", ".join(str(v) for v in motif_length_range),)
    print(f"best_motif_length: {best}")

    return 0


def _run_fit_k_elbow(
    args: argparse.Namespace,
    motiflets: "Motiflets",
    series: pd.Series,
) -> int:
    motif_length = args.motif_length
    motif_length_range: Optional[List[int]] = None

    if motif_length is None:
        motif_length_range = _resolve_motif_length_range(
            args.motif_length_range,
            len(series),
        )
        motif_length = motiflets.fit_motif_length(
            args.k_max,
            motif_length_range,
            subsample=args.subsample,
            plot=False
        )

    dists, motif_sets, elbow_points = motiflets.fit_k_elbow(
        args.k_max,
        motif_length=motif_length,
        filter=not args.no_filter,
        plot_elbows=False,
        plot_motifs_as_grid=False,
    )

    print(f"dataset:      {motiflets.ds_name}")
    print(f"observations: {len(series)}")
    print(f"motif_length: {motiflets.motif_length}")

    if motif_length_range:
        print("motif_length_candidates: " + ", ".join(
            str(v) for v in motif_length_range), )

    elbow_list = _convert_elbow_points(elbow_points)
    print(
        "elbow_points: "
        + (", ".join(str(k) for k in elbow_list) if elbow_list else "none"),
    )

    for k in elbow_list:
        motif = _safe_index(motif_sets, k)
        dist = _safe_index(dists, k)
        motif_repr = _format_motif_set(motif)
        dist_value = _format_distance(dist)
        print(f"k={k}: distance={dist_value} motif_set={motif_repr}")


    return 0


def _load_series(
    source: str,
    column: Optional[Union[str, int]],
    index_column: Optional[Union[str, int]],
    dropna: bool,
) -> pd.Series:
    read_kwargs = {}
    if index_column is not None:
        read_kwargs["index_col"] = index_column

    frame = _read_csv_with_errors(source, read_kwargs)
    if frame.empty:
        raise CLIError(f"no data found in '{source}'")

    series = _select_series(frame, column)
    if dropna:
        series = series.dropna()
    if series.empty:
        raise CLIError("time series contains no valid observations")

    return series


def _select_series(
    frame: pd.DataFrame,
    column: Optional[Union[str, int]],
) -> pd.Series:
    if column is None:
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] >= 1:
            raise CLIError(
                "cannot infer time series column; provide --column explicitly",
            )
        elif frame.shape[1] == 1:
            candidate = frame.iloc[:, 1]
    else:
        candidate = _locate_column(frame, column)

    series = candidate.squeeze()
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series.name = candidate.name
    return series


def _locate_column(
    frame: pd.DataFrame,
    column: Union[str, int],
) -> pd.Series:
    if isinstance(column, int):
        if column < 0 or column >= frame.shape[1]:
            raise CLIError(
                f"column index {column} out of bounds for dataframe with "
                f"{frame.shape[1]} columns",
            )
        return frame.iloc[:, column]

    if column not in frame.columns:
        raise CLIError(f"column '{column}' not found in data")

    return frame[column]


def _resolve_motif_length_range(
    raw_values: Optional[Sequence[str]],
    series_length: int,
) -> List[int]:
    if raw_values:
        resolved = _parse_motif_length_range(raw_values)
    else:
        resolved = _default_motif_length_range(series_length)

    if not resolved:
        raise CLIError("motif length range cannot be empty")

    deduped = sorted({value for value in resolved if value > 0})
    bounded = [value for value in deduped if value <= series_length]
    if not bounded:
        raise CLIError("motif length candidates exceed the series length")

    return bounded


def _read_csv_with_errors(source: str, read_kwargs: dict) -> pd.DataFrame:
    try:
        return pd.read_csv(source, **read_kwargs)
    except FileNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - pandas raises various errors
        raise CLIError(f"failed to read '{source}': {exc}") from exc


def _default_motif_length_range(series_length: int) -> List[int]:
    if series_length <= 0:
        raise CLIError("series length must be positive")

    minimum = max(4, series_length // 50)
    maximum = max(minimum + 1, series_length // 4)
    maximum = min(maximum, series_length)
    if minimum >= maximum:
        return [minimum]

    desired_points = 8
    step = max(1, (maximum - minimum) // (desired_points - 1))

    values: List[int] = []
    current = minimum
    while current <= maximum:
        values.append(current)
        current += step

    if values[-1] != maximum:
        values.append(maximum)

    return sorted(set(values))


def _convert_elbow_points(elbow_points) -> List[int]:
    if elbow_points is None:
        return []
    if isinstance(elbow_points, (list, tuple, set)):
        return sorted(int(k) for k in elbow_points)
    if hasattr(elbow_points, "tolist"):
        values = elbow_points.tolist()
        if isinstance(values, list):
            return sorted(int(k) for k in values)
    return [int(elbow_points)]


def _safe_index(container, index: int):
    try:
        return container[index]
    except Exception:  # pragma: no cover - defensive fallback
        return None


def _format_motif_set(motif) -> str:
    if motif is None:
        return "[]"
    if hasattr(motif, "tolist"):
        motif = motif.tolist()
    if isinstance(motif, (list, tuple)):
        return "[" + ", ".join(str(int(v)) for v in motif) + "]"
    return str(motif)


def _format_distance(distance) -> str:
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return "nan"
    if not pd.notna(value):
        return "nan"
    return f"{value:.6f}"


def _parse_column_identifier(value: Optional[str]) -> Optional[Union[str, int]]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _infer_dataset_name(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme and parsed.netloc:
        name = Path(parsed.path).stem
    else:
        name = Path(source).stem
    return name or "time_series"


def _discover_version() -> str:
    try:
        from motiflets import __version__
    except ImportError:  # pragma: no cover
        return "unknown"
    return __version__


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
