from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from motiflets import _cli


def _write_series(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=1024, freq="h"),
            "value": pd.Series(range(1024), dtype=float),
        }
    )
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    return path


def test_fit_motif_length_runs_pipeline(tmp_path, capsys):
    csv_path = _write_series(tmp_path)

    with mock.patch("motiflets._cli._get_motiflets_class") as get_class:
        motiflets_cls = mock.Mock()
        get_class.return_value = motiflets_cls

        instance = motiflets_cls.return_value
        instance.fit_motif_length.return_value = 128
        instance.ds_name = "series"

        exit_code = _cli.main(
            [
                "fit_motif_length",
                str(csv_path),
                "--k-max",
                "6",
                "--motif-length-range",
                "64",
                "128",
                "256",
            ]
        )

        assert exit_code == 0
        motiflets_cls.assert_called_once()
        instance.fit_motif_length.assert_called_once()

        args, kwargs = instance.fit_motif_length.call_args
        assert args[0] == 6
        assert args[1] == [64, 128, 256]
        assert kwargs["subsample"] == 2
        assert kwargs["plot"] is False

    captured = capsys.readouterr()
    assert "best_motif_length: 128" in captured.out
    assert "motif_length_candidates: 64, 128, 256" in captured.out


def test_fit_k_runs_pipeline(tmp_path, capsys):
    csv_path = _write_series(tmp_path)

    with mock.patch("motiflets._cli._get_motiflets_class") as get_class:
        motiflets_cls = mock.Mock()
        get_class.return_value = motiflets_cls

        instance = motiflets_cls.return_value
        instance.motif_length = 128
        instance.fit_k_elbow.return_value = None
        instance.dists = np.full((4, 1), np.inf)
        instance.dists[2, 0] = 0.5
        instance.motiflets = np.empty(4, dtype=object)
        instance.motiflets[2] = np.array([[10, 20]])
        instance.elbow_points = [np.array([2])]
        instance.ds_name = "series"

        exit_code = _cli.main(
            [
                "fit_k",
                str(csv_path),
                "--k-max",
                "5",
                "--motif-length",
                "128",
            ]
        )

        assert exit_code == 0
        instance.fit_motif_length.assert_not_called()
        instance.fit_k_elbow.assert_called_once()

        args, kwargs = instance.fit_k_elbow.call_args
        assert args[0] == 5
        assert kwargs["motif_length"] == 128
        assert kwargs["plot_elbows"] is False
        assert kwargs["plot_motifs_as_grid"] is False

    captured = capsys.readouterr()
    assert "motif_length: 128" in captured.out
    assert "k=2" in captured.out


def test_fit_k_cli_prints_duplicate_top_n_elbows(tmp_path, capsys):
    csv_path = _write_series(tmp_path)

    with mock.patch("motiflets._cli._get_motiflets_class") as get_class:
        motiflets_cls = mock.Mock()
        get_class.return_value = motiflets_cls

        instance = motiflets_cls.return_value
        instance.motif_length = 128
        instance.fit_k_elbow.return_value = None
        instance.dists = np.full((9, 2), np.inf)
        instance.dists[8, 0] = 0.8
        instance.dists[5, 0] = 0.5
        instance.dists[8, 1] = 0.9
        instance.motiflets = np.empty(9, dtype=object)
        instance.motiflets[8] = np.array([
            np.array([10, 20, 30, 40, 50, 60, 70, 80]),
            np.array([11, 21, 31, 41, 51, 61, 71, 81]),
        ], dtype=object)
        instance.motiflets[5] = np.array([
            np.array([100, 200, 300, 400, 500]),
            np.array([101, 201, 301, 401, 501]),
        ], dtype=object)
        instance.elbow_points = [np.array([8, 5]), np.array([8])]
        instance.ds_name = "series"

        exit_code = _cli.main(
            [
                "fit_k",
                str(csv_path),
                "--k-max",
                "8",
                "--motif-length",
                "128",
                "--top-n",
                "2",
            ]
        )

        assert exit_code == 0

    captured = capsys.readouterr()
    assert captured.out.count("k=8:") == 2
    assert "k=5:" in captured.out
    assert "motif_set=[10, 20, 30, 40, 50, 60, 70, 80]" in captured.out
    assert "motif_set=[11, 21, 31, 41, 51, 61, 71, 81]" in captured.out


def test_parse_motif_length_range_expands_colon_expressions():
    result = _cli._parse_motif_length_range(["2:6:2"])
    assert result == [2, 4, 6]


def test_parse_motif_length_range_invalid_step_raises():
    with pytest.raises(_cli.CLIError):
        _cli._parse_motif_length_range(["2:6:0"])


def test_resolve_motif_length_range_rejects_out_of_bounds():
    with pytest.raises(_cli.CLIError):
        _cli._resolve_motif_length_range(["100"], series_length=50)


def test_select_series_infers_numeric_column():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120, freq="h"),
            "value": pd.Series(range(120), dtype=float) * 0.01 + 1.0,
        }
    )

    series = _cli._select_series(frame, column=None)

    assert series.name == "value"
    assert len(series) == 120
    assert series.iloc[0] == pytest.approx(1.0)
    assert series.iloc[-1] == pytest.approx(1.0 + 0.01 * 119)


def test_locate_column_by_index_out_of_bounds_raises():
    frame = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(_cli.CLIError):
        _cli._locate_column(frame, 2)
