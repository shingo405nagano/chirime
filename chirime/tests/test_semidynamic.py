import datetime
import os
import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from chiriin.config import XY, XYZ, Delta, MeshDesign
from chiriin.semidynamic import SemiDynamic
from chiriin.web import (
    fetch_corrected_semidynamic_from_web,
    fetch_distance_and_azimuth_from_web,
)

test_prefecture_file = os.path.join(
    os.path.dirname(__file__), "data", "prefecture_pnt.csv"
)
# 47都道府県の主要な位置のデータを読み込む。
# このファイルの中には、Web版のAPIで取得した正しい補正後の経緯度も記録されている。
df = pd.read_csv(test_prefecture_file)


@pytest.mark.parametrize(
    "lon, lat, altitude",
    [
        (139.6917, 35.6895, 0.0),
        (135.5023, 34.6937, 10.0),
        ([139.6917, 135.5023], [35.6895, 34.6937], [0.0, 10.0]),
    ],
)
def test_convert_lon_lat_from_semidynamic(lon, lat, altitude):
    """Test the conversion of longitude and latitude using semidynamic correction."""
    semidynamic = SemiDynamic(lon, lat, datetime.datetime(2024, 4, 1, 0, 0, 0), altitude)


def test_read_parameters_from_semidynamic():
    """Test reading parameters from semidynamic correction."""
    semidynamic = SemiDynamic(
        139.6917, 35.6895, datetime.datetime(2024, 4, 1, 0, 0, 0), 0.0
    )
    df = semidynamic._read_parameters()
    assert isinstance(df, pd.DataFrame), "Returned value should be a DataFrame"


def test_adjust_mesh_code_from_semidynamic():
    """Test adjusting mesh code from semidynamic correction."""
    semidynamic = SemiDynamic(
        lon=139.6917,
        lat=35.6895,
        altitude=0.0,
        datetime_=datetime.datetime(2022, 4, 1, 0, 0, 0),
    )
    mesh_design = semidynamic._adjust_mesh_code(
        lower_left_sec_lon=float(semidynamic.lon * 3600),
        lower_left_sec_lat=float(semidynamic.lat * 3600),
    )
    assert isinstance(mesh_design, MeshDesign), (
        "Returned value should be a MeshDesign instance"
    )
    assert isinstance(mesh_design.lon, float)
    assert 0 < mesh_design.lon
    assert isinstance(mesh_design.lat, float)
    assert 0 < mesh_design.lat
    assert isinstance(mesh_design.standard_mesh_code, str)
    assert len(mesh_design.standard_mesh_code) == 8


@pytest.mark.parametrize(
    (
        "lon, lat, lower_left_mesh_code, lower_right_mesh_code, "
        "upper_left_mesh_code, upper_right_mesh_code"
    ),
    [
        (140.463488, 40.608410, "60407305", "60407400", "60407355", "60407450"),
        (141.344604, 43.063119, "64414255", "64414350", "64415205", "64415300"),
    ],
)
def test_mesh_design_from_semidynamic(
    lon,
    lat,
    lower_left_mesh_code,
    lower_right_mesh_code,
    upper_left_mesh_code,
    upper_right_mesh_code,
):
    """Test mesh design from semidynamic correction."""
    semidynamic = SemiDynamic(lon, lat, datetime.datetime(2024, 4, 1, 0, 0, 0), 0.0)
    mesh_design = semidynamic.mesh_design(lon, lat)
    assert isinstance(mesh_design, dict), (
        "Returned value should be a dictionary containing MeshDesign instances"
    )
    assert mesh_design["lower_left"].standard_mesh_code == lower_left_mesh_code
    assert mesh_design["lower_right"].standard_mesh_code == lower_right_mesh_code
    assert mesh_design["upper_left"].standard_mesh_code == upper_left_mesh_code
    assert mesh_design["upper_right"].standard_mesh_code == upper_right_mesh_code


@pytest.mark.parametrize(
    "mesh_code",
    [
        ("64415300"),
        (60407450),
        (60407305.0),
    ],
)
def test_get_delta_from_semidynamic(mesh_code):
    """Test getting delta values from semidynamic correction."""
    semidynamic = SemiDynamic(
        139.6917, 35.6895, datetime.datetime(2024, 4, 1, 0, 0, 0), 0.0
    )
    delta = semidynamic._get_delta(mesh_code)
    assert isinstance(delta, Delta), "Returned value should be a Delta instance"
    assert isinstance(delta.delta_x, Decimal)


def test_get_delta_sets_from_semidynamic():
    """Test that the delta values are correctly set from semidynamic correction."""
    semidynamic = SemiDynamic(
        139.6917, 35.6895, datetime.datetime(2024, 4, 1, 0, 0, 0), 0.0
    )
    mesh_design = semidynamic.mesh_design(float(semidynamic.lon), float(semidynamic.lat))
    delta_sets = semidynamic._get_delta_sets(mesh_design)
    assert isinstance(delta_sets, dict), "Returned value should be a dictionary"
    for delta in delta_sets.values():
        assert isinstance(delta, Delta), "Value should be a Delta instance"
        assert isinstance(delta.delta_x, Decimal)
        assert isinstance(delta.delta_y, Decimal)
        assert isinstance(delta.delta_z, Decimal)


def test_correction_2d_from_semidynamic():
    """
    Test the 2D correction from semidynamic correction.
    殆どの箇所で1cm未満の誤差だが、高知、長崎、熊本だけは、パラメーターファイルから
    メッシュコードの検索が失敗したので、ズレが2cm程度ある。
    """
    difference = []
    for _, row in df.iterrows():
        current_lon = row["longitude"]
        current_lat = row["latitude"]
        current_alti = row["altitude"]
        current_datetime = datetime.datetime.strptime(
            row["corrected_datetime"], "%Y-%m-%d %H:%M:%S"
        )
        return_to_original = row["return_to_original"]
        # セミダイナミック補正の実行
        semidynamic = SemiDynamic(
            lon=current_lon,
            lat=current_lat,
            altitude=current_alti,
            datetime_=current_datetime,
        )
        corrected_xy = semidynamic.correction_2d(return_to_original)
        while True:
            try:
                resps = fetch_distance_and_azimuth_from_web(
                    lons1=[corrected_xy.x],
                    lats1=[corrected_xy.y],
                    lons2=[row["corrected_lon"]],
                    lats2=[row["corrected_lat"]],
                    ellipsoid="bessel",
                )
                distance = resps[0]["distance"]
            except Exception:
                time.sleep(1)
            else:
                break
        # 5cm未満の誤差であることを確認
        assert distance <= 0.05
        difference.append(distance)
    # 平均的には1cm未満の誤差であることを確認
    assert np.mean(difference) < 0.01


def test_correction_2d_iterable_from_semidynamic():
    """Test the 2D correction with iterable inputs."""
    lon = df["longitude"].tolist()[:5]
    lat = df["latitude"].tolist()[:5]
    datetime_ = datetime.datetime(2024, 4, 1, 0, 0, 0)
    semidynamic = SemiDynamic(
        lon=lon,
        lat=lat,
        datetime_=datetime_,
    )
    corrected_xy = semidynamic.correction_2d(return_to_original=True)
    assert isinstance(corrected_xy, list), "Returned value should be a list"
    resps = fetch_corrected_semidynamic_from_web(
        lons=lon,
        lats=lat,
        correction_datetime=datetime_,
        return_to_original=True,
    )
    loop = True
    while loop:
        try:
            dist_list = fetch_distance_and_azimuth_from_web(
                lons1=[xy.x for xy in corrected_xy],
                lats1=[xy.y for xy in corrected_xy],
                lons2=[xyz.x for xyz in resps],
                lats2=[xyz.y for xyz in resps],
                ellipsoid="bessel",
            )
            distance_list = [resp["distance"] for resp in dist_list]
            if all([isinstance(v, float) for v in distance_list]):
                loop = False
        except Exception:
            time.sleep(1)
    assert np.mean(distance_list) < 0.01, "Average distance should be less than 0.01"


def test_correction_2d_web_api_from_semidynamic():
    """Test the 2D correction using web API."""
    lon = df["longitude"].tolist()[:5]
    lat = df["latitude"].tolist()[:5]
    altitude = df["altitude"].tolist()[:5]
    datetime_ = datetime.datetime(2024, 4, 1, 0, 0, 0)
    semidynamic = SemiDynamic(
        lon=lon,
        lat=lat,
        altitude=altitude,
        datetime_=datetime_,
    )
    resps = semidynamic.correction_2d_with_web_api(return_to_original=True)
    assert isinstance(resps, list), "Returned value should be a list"
    assert all(isinstance(res, XY) for res in resps), (
        "All items in the list should be XY instances"
    )
    semidynamic = SemiDynamic(
        lon=lon[0],
        lat=lat[0],
        altitude=altitude[0],
        datetime_=datetime_,
    )
    resps = semidynamic.correction_2d_with_web_api(return_to_original=False)
    assert isinstance(resps, XY), "Returned value should be an XY instance"


def test_correction_3d_from_semidynamic():
    """Test the 3D correction from semidynamic correction."""
    lon = df["longitude"].tolist()[:5]
    lat = df["latitude"].tolist()[:5]
    altitude = df["altitude"].tolist()[:5]
    datetime_ = datetime.datetime(2024, 4, 1, 0, 0, 0)
    semidynamic = SemiDynamic(
        lon=lon,
        lat=lat,
        altitude=altitude,
        datetime_=datetime_,
    )
    corrected_xyz = semidynamic.correction_3d_with_web_api(return_to_original=True)
    assert isinstance(corrected_xyz, list), "Returned value should be a list"
    assert all(isinstance(xyz, XYZ) for xyz in corrected_xyz), (
        "All items in the list should be XYZ instances"
    )
