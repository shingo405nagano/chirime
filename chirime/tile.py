import math
import time

import numpy as np
import pyproj
import requests
import shapely

from .config import XY, TileInfo, TileScope
from .formatter import (
    type_checker_crs,
    type_checker_float,
    type_checker_integer,
    type_checker_shapely,
    type_checker_zoom_level,
)
from .geometries import transform_geometry, transform_xy


def download_tile_array(url: str) -> np.ndarray:
    """
    ## Summary:
        地理院の標高APIを利用して、指定されたURLからタイルデータを取得する関数です。
        タイルデータは`bytes`型で返されるので、Float型の`np.ndarray`に変換して返しマス。
    """
    max_retries = 5
    retries = 0
    while True:
        if max_retries < retries:
            raise Exception(
                "Max retries exceeded, unable to download tile data. "
                f"\nRequest URL: {url}"
            )
        try:
            response = requests.get(url)
            if response.status_code != 200:
                retries += 1
                time.sleep(0.5)
                continue
            response_content = response.content
            # np.ndarrayに変換する処理を追加
            # ここでは、タイルデータがテキスト形式であることを前提としています。
            # もしバイナリ形式であれば、適切な変換方法を使用してください。
            tile_txt = response_content.decode("utf-8")
            # 'e'を'-9999'に置き換え、NaNに変換するための処理
            tile_data = tile_txt.replace("e", "-9999").splitlines()
            tile_data = [[float(v) for v in line.split(",")] for line in tile_data]
        except Exception as e:
            print(f"Error downloading tile: {e}")
            time.sleep(1)
        else:
            break
    ary = np.array(tile_data, dtype=np.float32)
    ary[ary == -9999] = np.nan  # -9999をNaNに変換
    return ary


@type_checker_integer(arg_index=0, kward="zoom_level")
@type_checker_zoom_level(arg_index=0, kward="zoom_level", min_zl=0, max_zl=24)
def cut_off_points(zoom_level: int) -> dict[str, list[float]]:
    """
    ## Summary:
        'zoom_level'で指定した地図タイルの座標を計算する。
        地図タイルは左上から計算されるので、取得出来る座標も左上のものです。
    Args:
        zoom_level (int):
            ズームレベルを指定する整数値。
            0-24の範囲にある整数で指定する必要があります。
    Returns:
        dict[str, list[float]]:
            タイルの左上の座標を表す辞書。座標は`Web Mercator`座標系（EPSG:3857）です。
            'X'キーには緯度、'Y'キーには経度のリストが含まれます。
    """
    web_mercator_scope = TileScope()
    x_length = web_mercator_scope.x_max - web_mercator_scope.x_min
    y_length = web_mercator_scope.y_max - web_mercator_scope.y_min
    side = 2**zoom_level
    X = [web_mercator_scope.x_min + i * (x_length / side) for i in range(side + 1)]
    Y = [web_mercator_scope.y_min + i * (y_length / side) for i in range(side + 1)]
    Y.sort(reverse=True)
    return {"X": X, "Y": Y}


@type_checker_float(arg_index=0, kward="lon")
@type_checker_float(arg_index=1, kward="lat")
@type_checker_integer(arg_index=2, kward="zoom_level")
@type_checker_crs(arg_index=3, kward="in_crs")
def lonlat_to_tile_idx(
    lon: float,  #
    lat: float,
    zoom_level: int,
    in_crs: str | int | pyproj.CRS,
) -> tuple[int, int]:
    """
    ## Summary:
        経緯度とズームレベルからタイルのインデックスを計算する関数。
    Args:
        lon (float):
        lat (float):
        zoom_level (int):
    Returns:
        tuple[int, int]:
            タイルのインデックス（x, y）
    """
    if in_crs.to_epsg() != 4326:
        # 入力座標系が経緯度でない場合、変換を行う
        xy = transform_xy(lon, lat, in_crs, "EPSG:4326")
    else:
        xy = XY(lon, lat)
    n = 2.0**zoom_level
    x_index = int((xy.x + 180.0) / 360.0 * n)
    _y = math.log(math.tan(math.radians(xy.y)) + 1 / math.cos(math.radians(xy.y)))
    y_index = int(n * (1 - _y / math.pi) / 2)
    return x_index, y_index


@type_checker_float(arg_index=0, kward="x")
@type_checker_float(arg_index=1, kward="y")
@type_checker_crs(arg_index=3, kward="in_crs")
def search_tile_info_from_xy(
    x: float,  #
    y: float,
    zoom_level: int,
    in_crs: str | int | pyproj.CRS,
    **kwargs,
) -> TileInfo:
    """
    ## Summary:
        指定した座標とズームレベルを含むタイルの情報を取得する。
        座標は"Iterable"な値を受け取らないので注意。基本的に取得するタイルのサイズは
        256x256ピクセルであることを前提としています。サイズが異なる場合は、
        `width`と`height`のキーワード引数を使用して指定できます。
    Args:
        x(float):
            x座標
        y(float):
            y座標
        zoom_level(int):
            ズームレベルを指定する整数値。
            0-24の範囲にある整数で指定する必要があります。
        in_crs(str | int | pyproj.CRS):
            入力座標系を指定するオプションの引数。
            指定しない場合は、経緯度（EPSG:4326）として解釈されます。
        **kwargs:
            - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
            - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
            - cut_off_points_lst(list[float]): 'cut_off_points'で取得した座標のリスト
            - x_idx(int): タイルのx座標
            - y_idx(int): タイルのy座標
    Returns:
        TileInfo:
            指定された座標とズームレベルに対応するタイルの情報を含むTileInfoオブジェクト。
            - x_idx(int): タイルのx座標
            - y_idx(int): タイルのy座標
            - zoom_level(int): ズームレベル
            - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                                    を含むTileScopeオブジェクト
            - x_resolution(float): タイルのx方向の解像度
            - y_resolution(float): タイルのy方向の解像度
            - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
            - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
            - in_crs(pyproj.CRS): 入力座標系を表すpyproj.CRSオブジェクト。
                                  デフォルトはEPSG:3857（Web Mercator）です。
    """
    # ズームレベルに対応するタイルの座標を取得
    tile_cds = kwargs.get("cut_off_points_lst")
    if tile_cds is None:
        tile_cds = cut_off_points(zoom_level)
    # タイルの"X"インデックスと"Y"インデックスを検索
    if "x_idx" in kwargs and "y_idx" in kwargs:
        x_idx = kwargs["x_idx"]
        y_idx = kwargs["y_idx"]
    else:
        if in_crs.to_epsg() != 3857:
            # 入力座標系がWeb Mercatorでない場合、変換を行う
            xy = transform_xy(x, y, in_crs, "EPSG:3857")
        else:
            xy = XY(x, y)
        x_idx, y_idx = lonlat_to_tile_idx(xy.x, xy.y, zoom_level, in_crs="EPSG:3857")
    # タイルの範囲を計算
    tile_scope = TileScope(
        x_min=tile_cds["X"][x_idx],
        y_min=tile_cds["Y"][y_idx + 1],
        x_max=tile_cds["X"][x_idx + 1],
        y_max=tile_cds["Y"][y_idx],
    )
    # タイルの解像度を計算
    width = kwargs.get("width", 256)
    height = kwargs.get("height", 256)
    x_resolution = (tile_scope.x_max - tile_scope.x_min) / width
    y_resolution = (tile_scope.y_max - tile_scope.y_min) / height
    return TileInfo(
        x_idx=x_idx,
        y_idx=y_idx,
        zoom_level=zoom_level,
        tile_scope=tile_scope,
        x_resolution=round(x_resolution, 4),
        y_resolution=round(y_resolution, 4),
    )


@type_checker_shapely(arg_index=0, kward="geometry")
@type_checker_integer(arg_index=1, kward="zoom_level")
@type_checker_crs(arg_index=2, kward="in_crs")
def search_tile_info_from_geometry(
    geometry: shapely.geometry.base.BaseGeometry,
    zoom_level: int,
    in_crs: str | pyproj.CRS,
    **kwargs,
) -> list[TileInfo]:
    """
    ## Summary:
        指定したジオメトリとズームレベルを含むタイルの情報を取得する。
        ジオメトリはshapelyのBaseGeometryオブジェクトで指定します。
        ジオメトリの範囲が複数のタイルにまたがる場合は、listで返されます。
        基本的に取得するタイルのサイズは256x256ピクセルであることを前提としていますが、
        サイズが異なる場合は、`width`と`height`のキーワード引数を使用して指定できます。
    Args:
        geometry(shapely.geometry.base.BaseGeometry):
            タイルを検索するためのジオメトリ。
        zoom_level(int):
            ズームレベルを指定する整数値。
            0-24の範囲にある整数で指定する必要があります。
        in_crs(Optional[str | pyproj.CRS]):
            入力座標系を指定するオプションの引数。
            指定しない場合は、経緯度（EPSG:4326）として解釈されます。
        **kwargs:
            - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
            - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
    Returns:
        TileInfo | list[TileInfo]:
            指定されたジオメトリとズームレベルに対応するタイルの情報を含むTileInfoオブジェクト。
    """
    if in_crs.to_epsg() != 3857:
        # 入力座標系がWeb Mercatorでない場合、変換を行う
        geometry = transform_geometry(geometry, in_crs, "EPSG:3857")
    # ジオメトリの範囲を取得
    geometry_scope = TileScope(*geometry.bounds)
    upper_left_xy = XY(geometry_scope.x_min, geometry_scope.y_max)
    lower_right_xy = XY(geometry_scope.x_max, geometry_scope.y_min)
    # ズームレベルに対応するタイルの座標を取得
    tile_cds = cut_off_points(zoom_level)
    # タイルの"X"インデックスと"Y"インデックスを検索
    x_min_idx, y_min_idx = lonlat_to_tile_idx(
        upper_left_xy.x, upper_left_xy.y, zoom_level, in_crs="EPSG:3857"
    )
    x_max_idx, y_max_idx = lonlat_to_tile_idx(
        lower_right_xy.x, lower_right_xy.y, zoom_level, in_crs="EPSG:3857"
    )
    # タイルのインデックスが一致しない場合は、その中間のタイルも考慮する
    tiles = []
    if x_min_idx == x_max_idx and y_min_idx == y_max_idx:
        # 単一のタイルに収まる場合
        tiles.append(
            search_tile_info_from_xy(
                upper_left_xy.x,
                upper_left_xy.y,
                zoom_level,
                in_crs=pyproj.CRS.from_epsg(3857),
                cut_off_points_lst=tile_cds,
                **kwargs,
            )
        )
    else:
        x_idxs, y_idxs = np.meshgrid(
            range(x_min_idx, x_max_idx + 1),  #
            range(y_min_idx, y_max_idx + 1),
        )
        for x_idx, y_idx in zip(x_idxs.flatten(), y_idxs.flatten(), strict=False):
            tile_info = search_tile_info_from_xy(
                0.0,  # x_idxで指定されるので、ここでは0.0を使用
                0.0,  # x_idxで指定されるので、ここでは0.0を使用
                zoom_level,
                in_crs=pyproj.CRS.from_epsg(3857),
                cut_off_points_lst=tile_cds,
                x_idx=x_idx,
                y_idx=y_idx,
                **kwargs,
            )
            tiles.append(tile_info)
    return tiles
