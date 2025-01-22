from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import requests
from pandas.testing import assert_frame_equal, assert_series_equal

from skrub.datasets import _fetching, _utils


@pytest.mark.parametrize(
    "dataset_name",
    [
        "medical_charge",
        "employee_salaries",
        # TODO
    ],
)
def test_fetching_single_tables(monkeypatch, dataset_name):
    with TemporaryDirectory() as temp_dir:
        fetch_func = getattr(_fetching, f"fetch_{dataset_name}")
        bunch = fetch_func(data_home=temp_dir)

        def _error_on_get(*args, **kwargs):
            raise AssertionError("request called")

        # Raise an error if requests.get() is called
        monkeypatch.setattr(requests, "get", _error_on_get)

        # calling the fetch function one more time
        local_bunch = fetch_func(data_home=temp_dir)

    assert_frame_equal(bunch.X, local_bunch.X)
    if hasattr(bunch, "y"):
        if isinstance(bunch.y, pd.Series):
            assert_series_equal(bunch.y, local_bunch.y)
        else:
            assert_frame_equal(bunch.y, local_bunch.y)
    assert bunch["metadata"] == local_bunch["metadata"]


def test_fetching_wrong_checksum(monkeypatch):
    dataset_info = _utils.DATASET_INFO["employee_salaries"]
    monkeypatch.setitem(dataset_info, "sha256", "bad_checksum")
    with pytest.raises(OSError, match=".*Can't download"):
        with TemporaryDirectory() as temp_dir:
            _fetching.fetch_employee_salaries(data_home=temp_dir)


def test_warning_redownload_checksum_has_changed(monkeypatch):
    with TemporaryDirectory() as temp_dir:
        _ = _fetching.fetch_employee_salaries(data_home=temp_dir)

        # altering the checksum to trigger a new download
        dataset_info = _utils.DATASET_INFO["employee_salaries"]
        monkeypatch.setitem(dataset_info, "sha256", "bad_checksum")

        with pytest.warns(match=r".*re-downloading.*"):
            # In the context of testing, altering the checksum will also raise an
            # error after re-downloading, which we doesn't want.
            try:
                _ = _fetching.fetch_employee_salaries(data_home=temp_dir)
            except OSError:
                pass


def test_cant_download(monkeypatch):
    def _error_on_get(*args, **kwargs):
        raise Exception("some error happened")

    # Raise an error when requests.get() is called
    monkeypatch.setattr(requests, "get", _error_on_get)

    with TemporaryDirectory() as temp_dir:
        with pytest.raises(OSError, match="Can't download"):
            _ = _fetching.fetch_employee_salaries(data_home=temp_dir)
