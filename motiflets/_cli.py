"""Command-line interface for motiflets."""

from __future__ import annotations

import warnings

from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", NumbaPerformanceWarning)

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union
from urllib.parse import urlparse

import pandas as pd

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
        dropna=not args.keep_na,
    )

    ds_name = args.ds_name or _infer_dataset_name(args.source)
    motiflets = Motiflets(
        ds_name=ds_name,
        series=series,
        distance=args.distance,
        n_jobs=args.n_jobs
    )

    print(f"Loaded dataset '{motiflets.ds_name}' with {len(series)} observations.")

    if args.command == "fit_motif_length":
        return _run_fit_motif_length(args, motiflets, series)
    if args.command == "fit_k":
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
        "--keep-na",
        action="store_true",
        help="Keep missing values instead of dropping them.",
    )
    common.add_argument(
        "--distance",
        default="znormed_ed",
        help="Distance metric name (default: znormed_ed).",
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
        required=True,
        help="Search k in [2, k_max] (required).",
    )
    fit_motif_length.add_argument(
        "--motif-length-range",
        nargs="+",
        required=True,
        help="Explicit motif length candidates (e.g. 24:192:24).",
    )
    fit_motif_length.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsampling factor to speed up computations (default: 2).",
    )

    fit_k_elbow = subparsers.add_parser(
        "fit_k",
        parents=[common],
        help="Search motif sets across different k.",
    )
    fit_k_elbow.add_argument(
        "--k-max",
        type=int,
        required=True,
        help="Search k in [2, k_max] (required).",
    )
    fit_k_elbow.add_argument(
        "--motif-length",
        type=int,
        required=True,
        help="Use a fixed motif length (required).",
    )
    fit_k_elbow.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsampling factor to speed up automatic length search (default: 2).",
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
    print(f"motif_length_candidates: " + ", ".join(str(v) for v in motif_length_range))
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
        raise CLIError(f"motif-length parameter must be set.")

    dists, motif_sets, elbow_points = motiflets.fit_k_elbow(
        args.k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifs_as_grid=False,
    )

    print(f"dataset:      {motiflets.ds_name}")
    print(f"observations: {len(series)}")
    print(f"motif_length: {motiflets.motif_length}")

    if motif_length_range:
        print("motif_length_candidates: " + ", ".join(
            str(v) for v in motif_length_range), )

    elbow_list = list(elbow_points)
    print(
        "elbow_points: "
        + (", ".join(str(k) for k in elbow_list) if elbow_list else "none"),
    )

    for k in elbow_list:
        motif = motif_sets[k]
        dist = dists[k]
        motif_repr = _format_motif_set(motif)
        dist_value = _format_distance(dist)
        print(f"k={k}: distance={dist_value} motif_set={motif_repr}")

    return 0


def _load_series(
        source: str,
        column: Optional[Union[str, int]],
        dropna: bool,
) -> pd.Series:
    read_kwargs = {}

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
        if frame.shape[1] == 2:
            candidate = frame.iloc[:, 1]
        elif numeric.shape[1] == 1:
            candidate = frame.iloc[:, 0]
        elif numeric.shape[1] >= 1:
            raise CLIError(
                "cannot infer time series column; provide --column explicitly",
            )
    else:
        candidate = _locate_column(frame, column)

    series = candidate.squeeze()
    if not isinstance(series, pd.Series):
        series = pd.Series(np.ascontiguousarray(series))
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

    min_len = max(4, series_length // 50)
    max_len = min(series_length // 4, series_length)
    if min_len >= max_len:
        return [min_len]

    desired_points = 8
    step = max(1, (max_len - min_len) // (desired_points - 1))
    values = list(range(min_len, max_len + 1, step))

    if values[-1] != max_len:
        values.append(max_len)

    return sorted(set(values))


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


def _parse_motif_length_range(raw_values: Sequence[str]) -> List[int]:
    if not raw_values:
        return []

    candidates: List[int] = []
    for token in raw_values:
        value = token.strip()
        if not value:
            raise CLIError("motif length specification cannot be empty")

        if ":" in value:
            parts = [part.strip() for part in value.split(":")]
            if len(parts) not in (2, 3):
                raise CLIError(
                    f"invalid motif length range expression '{value}' "
                    "(expected start:stop[:step])",
                )

            try:
                start = int(parts[0])
                stop = int(parts[1])
                step = int(parts[2]) if len(parts) == 3 else 1
            except ValueError as exc:
                raise CLIError(
                    f"invalid motif length range expression '{value}'",
                ) from exc

            if step == 0:
                raise CLIError("motif length step cannot be zero")

            increasing = start <= stop
            if (step > 0 and not increasing) or (step < 0 and increasing):
                raise CLIError(
                    f"range '{value}' does not progress towards the stop value",
                )

            current = start
            if step > 0:
                while current <= stop:
                    candidates.append(current)
                    current += step
            else:
                while current >= stop:
                    candidates.append(current)
                    current += step
            continue

        try:
            candidates.append(int(value))
        except ValueError as exc:
            raise CLIError(f"invalid motif length '{value}'") from exc

    return candidates


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
