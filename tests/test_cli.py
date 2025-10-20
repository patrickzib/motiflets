from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import pandas as pd
import pandas.testing as pdt
from urllib.error import HTTPError

from motiflets import _cli


def _write_series(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8, freq="MS"),
            "value": [1.0, 1.5, 1.2, 1.8, 2.0, 2.4, 2.2, 2.5],
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
        instance.fit_motif_length.return_value = 24
        instance.ds_name = "series"

        exit_code = _cli.main(
            ["fit_motif_length", str(csv_path), "--k-max", "6"]
        )

        assert exit_code == 0
        motiflets_cls.assert_called_once()
        instance.fit_motif_length.assert_called_once()

        args, kwargs = instance.fit_motif_length.call_args
        assert args[0] == 6
        assert len(args[1]) >= 1
        assert kwargs["subsample"] == 2

    captured = capsys.readouterr()
    assert "best_motif_length: 24" in captured.out
    assert "dataset: series" in captured.out


def test_fit_k_elbow_with_fixed_length(tmp_path, capsys):
    csv_path = _write_series(tmp_path)

    with mock.patch("motiflets._cli._get_motiflets_class") as get_class:
        motiflets_cls = mock.Mock()
        get_class.return_value = motiflets_cls

        instance = motiflets_cls.return_value
        instance.motif_length = 24
        instance.fit_k_elbow.return_value = (
            [None, None, 0.5, 0.8],
            [None, None, [10, 20], [15, 35, 55]],
            [2, 3],
        )
        instance.ds_name = "series"

        exit_code = _cli.main(
            [
                "fit_k_elbow",
                str(csv_path),
                "--motif-length",
                "24",
                "--k-max",
                "6",
            ]
        )

        assert exit_code == 0
        instance.fit_motif_length.assert_not_called()
        instance.fit_k_elbow.assert_called_once()

    captured = capsys.readouterr()
    assert "k=2" in captured.out
    assert "k=3" in captured.out
    assert "motif_length: 24" in captured.out


def test_fit_k_elbow_triggers_length_search(tmp_path):
    csv_path = _write_series(tmp_path)

    with mock.patch("motiflets._cli._get_motiflets_class") as get_class:
        motiflets_cls = mock.Mock()
        get_class.return_value = motiflets_cls

        instance = motiflets_cls.return_value
        instance.fit_motif_length.return_value = 30
        instance.motif_length = 30
        instance.fit_k_elbow.return_value = (
            [None, None, 0.4],
            [None, None, [5, 15]],
            [2],
        )
        instance.ds_name = "series"

        exit_code = _cli.main(["fit_k_elbow", str(csv_path), "--k-max", "4"])

        assert exit_code == 0
        instance.fit_motif_length.assert_called_once()
        instance.fit_k_elbow.assert_called_once()

        args, kwargs = instance.fit_motif_length.call_args
        assert args[0] == 4
        assert kwargs["subsample"] == 2


def test_load_series_uses_package_dataset(monkeypatch):
    raw_series = pd.Series([1.0, None, 2.0], index=[0, 1, 2], name="value")

    with_index = mock.Mock(return_value=raw_series)
    plain_loader = mock.Mock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr(_cli.ml, "read_dataset_with_index", with_index)
    monkeypatch.setattr(_cli.ml, "read_dataset", plain_loader)

    result = _cli._load_series(
        "AirPassengers.csv",
        column=None,
        index_column=None,
        dropna=True,
    )

    expected = raw_series.dropna()
    pdt.assert_series_equal(result, expected)
    with_index.assert_called_once_with("AirPassengers.csv")
    plain_loader.assert_not_called()


def test_load_series_fetches_remote_csv(monkeypatch):
    csv_bytes = b"value\n1\n2\n"

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    captured_request = {}

    calls = []

    def fake_urlopen(request):
        calls.append(request)
        captured_request["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        return FakeResponse(csv_bytes)

    monkeypatch.setattr(
        _cli.ml,
        "read_dataset_with_index",
        mock.Mock(side_effect=FileNotFoundError),
    )
    monkeypatch.setattr(
        _cli.ml,
        "read_dataset",
        mock.Mock(side_effect=FileNotFoundError),
    )
    monkeypatch.setattr(_cli.urllib.request, "urlopen", fake_urlopen)

    result = _cli._load_series(
        "https://example.com/data.csv",
        column=None,
        index_column=None,
        dropna=False,
    )

    pdt.assert_series_equal(result, pd.Series([1, 2], name="value"))
    assert "user-agent" in captured_request["headers"]
    assert len(calls) == 1


def test_fetch_url_bytes_retries_on_403(monkeypatch):
    csv_bytes = b"value\n1\n"

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    calls = []

    def fake_urlopen(request):
        calls.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                403,
                "Forbidden",
                hdrs=None,
                fp=io.BytesIO(b""),
            )
        return FakeResponse(csv_bytes)

    monkeypatch.setattr(_cli.urllib.request, "urlopen", fake_urlopen)

    data = _cli._fetch_url_bytes("https://example.com/data.csv")

    assert data == csv_bytes
    assert len(calls) == 2
    assert "referer" in calls[1]
