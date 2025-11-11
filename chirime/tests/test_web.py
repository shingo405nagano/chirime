import datetime
import os
import time

import numpy as np
import pandas as pd
import pytest

from chiriin.web import (
    fetch_corrected_semidynamic_from_web,
    fetch_elevation_from_web,
    fetch_elevation_tiles_from_web,
    fetch_geoid_height_from_web,
    fetch_img_map_tiles_from_web,
)

test_prefecture_file = os.path.join(
    os.path.dirname(__file__), "data", "prefecture_pnt.csv"
)
df = pd.read_csv(test_prefecture_file)
LON = df["longitude"].tolist()[:5]
LAT = df["latitude"].tolist()[:5]
ALTITUDE = df["altitude"].tolist()[:5]


def test_fetch_elevation_from_web():
    altitude_list = fetch_elevation_from_web(LON, LAT)
    assert isinstance(altitude_list, list)
    assert len(altitude_list) == len(LON)
    assert -100 < min(altitude_list) < 10000
    with pytest.raises(Exception):
        altitude = fetch_elevation_from_web([340.00], [40.00])
        assert isinstance(altitude, float)


@pytest.mark.parametrize(
    "datetime_, dimension, return_to_original, err",
    [
        (datetime.datetime(2023, 10, 1, 0, 0, 0), 2, True, False),
        (datetime.datetime(2022, 10, 1, 0, 0, 0), 2, False, False),
        (datetime.datetime(2021, 10, 1, 0, 0, 0), 3, True, False),
        (datetime.datetime(2020, 10, 1, 0, 0, 0), 0, True, True),
    ],
)
def test_fetch_corrected_semidynamic_from_web(
    datetime_, dimension, return_to_original, err
):
    # APIの制限により、連続してリクエストを送るとエラーになるため、テスト間に待機時間を設ける。
    time.sleep(10)
    corrected_xyz = fetch_corrected_semidynamic_from_web(
        datetime_,
        LON if err is False else [340.00],
        LAT if err is False else [40.00],
        ALTITUDE if err is False else [0.0],
        dimension=dimension,
        return_to_original=return_to_original,
    )
    if err is False:
        assert isinstance(corrected_xyz, list)
        assert len(corrected_xyz) == len(LON)
        for coords in corrected_xyz:
            assert isinstance(coords.x, float)
            assert isinstance(coords.y, float)
            assert isinstance(coords.z, float)
            if dimension == 2:
                assert coords.z == 0.0
            else:
                assert coords.z != 0.0
    else:
        for coords in corrected_xyz:
            assert coords is None


def test_fetch_elevation_tiles_from_web():
    urls = [
        "https://cyberjapandata.gsi.go.jp/xyz/dem/14/14569/6169.txt",
        "https://cyberjapandata.gsi.go.jp/xyz/dem/14/14569/6170.txt",
        "https://cyberjapandata.gsi.go.jp/xyz/dem/14/14569/6171.txt",
        "https://cyberjapandata.gsi.go.jp/xyz/dem/14/14569/6172.txt",
    ]
    resps = fetch_elevation_tiles_from_web(urls)
    assert isinstance(resps, dict)
    for _, ary in resps.items():
        assert isinstance(ary, np.ndarray)
        assert ary.shape == (256, 256)
        assert ary.dtype == "float32"


@pytest.mark.parametrize(
    "url, success",
    [
        ("https://cyberjapandata.gsi.go.jp/xyz/std/10/912/388.png", True),
        ("https://cyberjapandata.gsi.go.jp/xyz/std/15/29197/12432.png", True),
        ("https://cyberjapandata.gsi.go.jp/xyz/std/18/233577/99460.png", True),
        ("https://cyberjapandata.gsi.go.jp/xyz/std/19/467154/198921.png", False),
        ("https://cyberjapandata.gsi.go.jp/xyz/slopemap/10/912/388.png", True),
        ("https://cyberjapandata.gsi.go.jp/xyz/slopemap/15/29197/12432.png", True),
        ("https://cyberjapandata.gsi.go.jp/xyz/slopemap/18/233577/99460.png", False),
        ("https://cyberjapandata.gsi.go.jp/xyz/slopemap/19/467154/198921.png", False),
        ("https://www.google.com", False),
    ],
)
def test_fetch_img_map_tiles_from_web(url, success):
    resps = fetch_img_map_tiles_from_web([url])
    assert isinstance(resps, dict)
    if success:
        assert url in resps
        r = resps[url]
        assert isinstance(r, np.ndarray)
    else:
        # URLが存在しない場合、Noneが返されることを確認
        assert resps[url] is None


def test_fetch_geoid_height_from_web():
    x = [139.6917, 139.6917, 139.6917]
    y = [35.6895, 35.6895, 35.6895]
    resps = fetch_geoid_height_from_web(x, y)
    assert isinstance(resps, list)
    assert len(resps) == len(x)
    for resp in resps:
        assert isinstance(resp, float)
        assert 0 < resp

    resps = fetch_geoid_height_from_web([140.00], [40.00])
    assert isinstance(resps, float)
