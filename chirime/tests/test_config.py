import datetime
import os

import pandas as pd
import pytest
import requests

from chiriin.config import (
    ChiriinWebApi,
    SemidynamicCorrectionFiles,
    semidynamic_correction_file,
)


@pytest.mark.parametrize(
    "datetime_", [datetime.datetime(year, 4, 1, 0, 0, 0) for year in range(2010, 2026)]
)
def test_get_file_path_from_semidynamic_correction_files(datetime_):
    """
    Test that the file path is correctly retrieved for a given datetime.
    """
    semidynamic_files = SemidynamicCorrectionFiles()
    file_path = semidynamic_files._get_file_path(datetime_)
    assert os.path.exists(file_path), (
        f"File path {file_path} does not exist for datetime {datetime_}"
    )
    assert os.path.basename(file_path).endswith(f"{datetime_.year}.par"), (
        f"File path {file_path} does not match expected pattern for year {datetime_.year}"
    )


@pytest.mark.parametrize(
    "line",
    [
        ["4.5", "5.5", "6.5"],
        ["100", "200", "300"],
        ["1000", "", "\n", "2000", "3000\n"],
        ["", "", "\n", ""],
    ],
)
def test_clean_line_from_semidynamic_correction_files(line):
    """Test that the line is cleaned correctly."""
    semidynamic_files = SemidynamicCorrectionFiles()
    cleaned_line = semidynamic_files._clean_line(line)
    assert isinstance(cleaned_line, list), "Cleaned line should be a list"
    if not cleaned_line:
        pass
    assert all(isinstance(item, (int, float)) for item in cleaned_line), (
        "Cleaned line should contain only numbers (int or float)"
    )
    header = ["MeshCode", "dB(sec)", "dL(sec)", "dH(m)"]
    cleaned_line = semidynamic_files._clean_line(header)
    assert len(cleaned_line) == 4, "Header should have 4 elements"
    assert all(isinstance(item, str) for item in cleaned_line), (
        "Header should contain strings"
    )


@pytest.mark.parametrize(
    "datetime_", [datetime.datetime(year, 4, 1, 0, 0, 0) for year in range(2010, 2026)]
)
def test_semidynamic_correction_file(datetime_):
    """
    Test that the semidynamic correction file is correctly retrieved
    for a given datetime.
    """
    df = semidynamic_correction_file(datetime_)
    assert isinstance(df, pd.DataFrame), "Returned value should be a DataFrame"
    if datetime_.year == 2020:
        with pytest.raises(Exception):  # noqa: B017
            semidynamic_correction_file(datetime_.replace(year=2000, month=3))


def test_elevation_url_from_chiriin_web_api():
    """Test that the elevation URL is correctly formatted."""
    api = ChiriinWebApi()
    url = api.elevation_url()
    assert isinstance(url, str), "URL should be a string"
    resps = requests.get(url)
    assert isinstance(resps.json(), dict), "Response should be a dictionary"


def test_get_geoid_height_2011_url_from_chiriin_web_api():
    """Test that the geoid height URL is correctly formatted."""
    api = ChiriinWebApi()
    url = api.geoid_height_2011_url()
    assert isinstance(url, str), "URL should be a string"
    resps = requests.get(url.format(lon=139.6917, lat=35.6895))
    assert isinstance(resps.json(), dict), "Response should be a dictionary"


def test_get_geoid_height_2024_url_from_chiriin_web_api():
    """Test that the geoid height 2024 URL is correctly formatted."""
    api = ChiriinWebApi()
    url = api.geoid_height_2024_url()
    assert isinstance(url, str), "URL should be a string"
    resps = requests.get(url.format(lon=139.6917, lat=35.6895))
    assert isinstance(resps.json(), dict), "Response should be a dictionary"


def test_distance_and_azimuth_url_from_chiriin_web_api():
    """Test that the distance and azimuth URL is correctly formatted."""
    api = ChiriinWebApi()
    url = api.distance_and_azimuth_url().format(
        ellipsoid="GRS80", lon1=139.6917, lat1=35.6895, lon2=139.7000, lat2=35.7000
    )
    assert isinstance(url, str), "URL should be a string"
    resps = requests.get(url)
    assert isinstance(resps.json(), dict), "Response should be a dictionary"


def test_semidynamic_correction_url_from_chiriin_web_api():
    """Test that the semidynamic correction URL is correctly formatted."""
    api = ChiriinWebApi()
    url = api.semidynamic_correction_url().format(
        year=2024, sokuchi=1, dimension=2, lon=139.6917, lat=35.6895, alti=0.0
    )
    assert isinstance(url, str), "URL should be a string"
    resps = requests.get(url)
    assert isinstance(resps.json(), dict), "Response should be a dictionary"
