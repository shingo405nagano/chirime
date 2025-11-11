"""
1. "10進法経緯度"から"度分秒経緯度"への変換
2. "度分秒経緯度"から"10進法経緯度"への変換
3. どちらの経緯度でも、求める単位に変換（リストも可）
"""

import math
from decimal import Decimal
from typing import Iterable

import pyproj
import shapely
from shapely.geometry.base import BaseGeometry

from .config import XY, Scope
from .formatter import type_checker_crs, type_checker_float, type_checker_shapely
from .utils import dimensional_count


def dms_to_degree(
    dms: float,  #
    digits: int = 9,
    decimal_obj: bool = False,
) -> float | Decimal:
    """
    ## Description:
        度分秒経緯度を10進法経緯度に変換する関数
    Args:
        dms (float):
            度分秒経緯度
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            10進法経緯度をDecimal型で返すかどうか
    Returns:
        float | Decimal:
            10進法経緯度
    """
    try:
        dms = float(dms)
    except ValueError as err:
        raise ValueError("dms must be a float or convertible to float.") from err
    dms_txt = str(dms)
    sep = "."
    integer_part, decimal_part = dms_txt.split(sep)
    micro_sec = float(f"0.{decimal_part}")
    if len(integer_part) < 6 or 7 < len(integer_part):
        raise ValueError(f"dms must have a 6- or 7-digit integer part. Arg: {dms}")
    sec = Decimal(f"{(int(integer_part[-2:]) + micro_sec) / 3600}")
    min_ = Decimal(f"{int(integer_part[-4:-2]) / 60}")
    deg = Decimal(f"{float(int(integer_part[:-4]))}")
    if decimal_obj:
        return round(deg + min_ + sec, digits)
    return float(round(deg + min_ + sec, digits))


def degree_to_dms(
    degree: float,  #
    digits: int = 5,
    decimal_obj: bool = False,
) -> float | Decimal:
    """
    ## Description:
        10進法経緯度を度分秒経緯度に変換する関数
    Args:
        degree (float):
            10進法経緯度
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            度分秒経緯度をDecimal型で返すかどうか
    Returns:
        float | Decimal:
            度分秒経緯度
    """
    try:
        _degree = float(degree)
    except ValueError as err:
        # 10進法経緯度はfloatに変換可能な値である必要がある
        raise ValueError("degree must be a float or convertible to float.") from err
    if not (-180 <= _degree <= 180):
        # 経度は-180から180の範囲である必要がある
        raise ValueError(f"degree must be in the range of -180 to 180. Arg: {degree}")

    deg = int(degree)
    min_ = int((degree - deg) * 60)
    _sec = str((degree - deg - min_ / 60) * 3600)
    idx = _sec.find(".")
    sec = int(_sec[:idx] if idx != -1 else _sec)
    # マイクロ秒は小数点以下5桁までを想定
    micro_sec = int(round(int(_sec[idx + 1 :][: digits + 1]), digits) * 0.1)
    # 度分秒が10未満の場合は0を付与
    deg = f"0{deg}" if deg < 10 else str(deg)
    min_ = f"0{min_}" if min_ < 10 else str(min_)
    sec = f"0{sec}" if sec < 10 else str(sec)
    dms = float(f"{deg}{min_}{sec}.{micro_sec}")
    if decimal_obj:
        return Decimal(f"{dms:.{digits}f}")
    return dms


def _dms_to_degree_lonlat(
    lon: float,  #
    lat: float,
    digits: int = 9,
    decimal_obj: bool = False,
) -> XY:
    """
    ## Description:
        度分秒経緯度を10進法経緯度に変換する関数
    Args:
        lon (float):
            度分秒経緯度
        lat (float):
            度分秒経緯度
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            Decimal型で返すかどうか
    Returns:
        XY(NamedTuple):
            10進法経緯度
            - x: float | Decimal
            - y: float | Decimal
    Example:
        >>> dms_to_degree_lonlat(140516.27814, 36103600.00000)
        (140.087855042, 36.103774792)
    """
    deg_lon = dms_to_degree(lon, digits, decimal_obj)
    deg_lat = dms_to_degree(lat, digits, decimal_obj)
    return XY(x=deg_lon, y=deg_lat)


def _dms_to_degree_lonlat_list(
    lon_list: Iterable[float],
    lat_list: Iterable[float],
    decimal_obj: bool = False,
    digits: int = 9,
) -> list[XY]:
    """
    ## Description:
        リスト化された度分秒経緯度を10進法経緯度に変換する関数
    Args:
        lon_list (Iterable[float]):
            度分秒経緯度のリスト（経度）
        lat_list (Iterable[float]):
            度分秒経緯度のリスト（緯度）
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            Decimal型で返すかどうか
    Returns:
        list[XY(NamedTuple)]:
            10進法経緯度のリスト
            - x: float | Decimal
            - y: float | Decimal
    Example:
        >>> dms_to_degree_lonlat_list([140516.27814, 140516.27814], [36103600.00000, 36103600.00000])
        [(140.087855042, 36.103774792), (140.087855042, 36.103774792)]
    """
    assert len(lon_list) == len(lat_list), (
        "lon_list and lat_list must have the same length."
    )
    assert dimensional_count(lon_list) == 1, "lon_list must be a 1-dimensional iterable."
    assert dimensional_count(lat_list) == 1, "lat_list must be a 1-dimensional iterable."
    result = []
    for lon, lat in zip(lon_list, lat_list, strict=False):
        xy = _dms_to_degree_lonlat(lon, lat, digits, decimal_obj)
        result.append(xy)
    return result


def dms_to_degree_lonlat(
    lon: float | Iterable[float],
    lat: float | Iterable[float],
    digits: int = 9,
    decimal_obj: bool = False,
) -> XY | list[XY]:
    """
    ## Description:
        度分秒経緯度を10進法経緯度に変換する関数
    Args:
        lon (float | Iterable[float]):
            度分秒経緯度（経度）
        lat (float | Iterable[float]):
            度分秒経緯度（緯度）
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            Decimal型で返すかどうか
    Returns:
        XY | list[XY]:
            10進法経緯度
            - x: float | Decimal
            - y: float | Decimal
    """
    if isinstance(lon, Iterable):
        return _dms_to_degree_lonlat_list(lon, lat, decimal_obj, digits)
    return _dms_to_degree_lonlat(lon, lat, digits, decimal_obj)


def _degree_to_dms_lonlat(
    lon: float,  #
    lat: float,
    digits: int = 5,
    decimal_obj: bool = False,
) -> XY:
    """
    ## Description:
        10進法経緯度を度分秒経緯度に変換する関数
    Args:
        lon (float):
            10進法経緯度（経度）
        lat (float):
            10進法経緯度（緯度）
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            Decimal型で返すかどうか
    Returns:
        XY(NamedTuple):
            度分秒経緯度
            - x: float | Decimal
            - y: float | Decimal
    Example:
        >>> degree_to_dms_lonlat(140.08785504166664, 36.103774791666666)
        (140516.2781, 36103600.0000)
    """
    dms_lon = degree_to_dms(lon, digits, decimal_obj)
    dms_lat = degree_to_dms(lat, digits, decimal_obj)
    return XY(x=dms_lon, y=dms_lat)


def _degree_to_dms_lonlat_list(
    lon_list: Iterable[float],
    lat_list: Iterable[float],
    digits: int = 5,
    decimal_obj: bool = False,
) -> list[XY]:
    """
    ## Description:
        リスト化された10進法経緯度を度分秒経緯度に変換する関数
    Args:
        lon_list (Iterable[float]):
            10進法経緯度のリスト（経度）
        lat_list (Iterable[float]):
            10進法経緯度のリスト（緯度）
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            Decimal型で返すかどうか
    Returns:
        list[XY(NamedTuple)]:
            度分秒経緯度のリスト
            - x: float | Decimal
            - y: float | Decimal
    Example:
        >>> degree_to_dms_lonlat_list([140.08785504166664, 140.08785504166664], [36.103774791666666, 36.103774791666666])
        [(140516.2781, 36103600.0000), (140516.2781, 36103600.0000)]
    """
    assert len(lon_list) == len(lat_list), (
        "lon_list and lat_list must have the same length."
    )
    assert dimensional_count(lon_list) == 1, "lon_list must be a 1-dimensional iterable."
    assert dimensional_count(lat_list) == 1, "lat_list must be a 1-dimensional iterable."
    result = []
    for lon, lat in zip(lon_list, lat_list, strict=False):
        xy = _degree_to_dms_lonlat(lon, lat, digits, decimal_obj)
        result.append(xy)
    return result


def degree_to_dms_lonlat(
    lon: float | Iterable[float],
    lat: float | Iterable[float],
    digits: int = 5,
    decimal_obj: bool = False,
) -> XY | list[XY]:
    """
    ## Description:
        10進法経緯度を度分秒経緯度に変換する関数
    Args:
        lon (float | Iterable[float]):
            10進法経緯度（経度）
        lat (float | Iterable[float]):
            10進法経緯度（緯度）
        digits (int):
            小数点以下の桁数
        decimal_obj (bool):
            Decimal型で返すかどうか
    Returns:
        XY | list[XY]:
            度分秒経緯度
            - x: float | Decimal
            - y: float | Decimal
    """
    if isinstance(lon, Iterable):
        return _degree_to_dms_lonlat_list(lon, lat, digits, decimal_obj)
    return _degree_to_dms_lonlat(lon, lat, digits, decimal_obj)


@type_checker_float(arg_index=0, kward="x")
@type_checker_float(arg_index=1, kward="y")
@type_checker_crs(arg_index=2, kward="in_crs")
@type_checker_crs(arg_index=3, kward="out_crs")
def transform_xy(
    x: float | Iterable[float],  #
    y: float | Iterable[float],
    in_crs: str | int | pyproj.CRS,
    out_crs: str | int | pyproj.CRS,
) -> XY | list[XY]:
    """
    ## Summary:
        x座標とy座標を指定した座標系から別の座標系に変換する。
    Args:
        x (float | Iterable[float]):
            変換するx座標。単一の値または値のリスト。
        y (float | Iterable[float]):
            変換するy座標。単一の値または値のリスト。
        in_crs (str | int | pyproj.CRS):
            入力座標系。EPSGコードやCRSオブジェクトを指定。
        out_crs (str | int | pyproj.CRS):
            出力座標系。EPSGコードやCRSオブジェクトを指定。
    Returns:
        XY | list[XY]:
            変換後の座標。単一のXYオブジェクトまたはXYオブジェクトのリスト。
    """
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    if isinstance(x, Iterable):
        return [XY(x=lon_i, y=lat_i) for lon_i, lat_i in zip(lon, lat, strict=False)]
    return XY(x=lon, y=lat)


@type_checker_shapely(arg_index=0, kward="geometry")
@type_checker_crs(arg_index=1, kward="in_crs")
@type_checker_crs(arg_index=2, kward="out_crs")
def transform_geometry(
    geometry: BaseGeometry,
    in_crs: str | int | pyproj.CRS,
    out_crs: str | int | pyproj.CRS,
) -> BaseGeometry:
    """
    ## Summary:
        指定した座標系から別の座標系にジオメトリを変換する。
    Args:
        geometry (BaseGeometry):
            変換するジオメトリ。shapelyのBaseGeometryオブジェクト。
        in_crs (str | int | pyproj.CRS):
            入力座標系。EPSGコードやCRSオブジェクトを指定。
        out_crs (str | int | pyproj.CRS):
            出力座標系。EPSGコードやCRSオブジェクトを指定。
    Returns:
        BaseGeometry:
            変換後のジオメトリ。shapelyのBaseGeometryオブジェクト。
    """
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    try:
        # shapelyのバージョンにより、transformの使い方が異なるため、両方試す
        transformed_geom = shapely.transform(  # type: ignore
            geometry, transformer.transform, interleaved=False
        )
    except Exception:
        try:
            transformed_geom = shapely.ops.transform(transformer.transform, geometry)
        except ValueError as e:  # noqa: E722
            raise ValueError(f"Failed to transform geometry: {e}") from e
    return transformed_geom


@type_checker_float(arg_index=0, kward="lon")
@type_checker_float(arg_index=1, kward="lat")
def estimate_utm_crs(lon: float, lat: float, datum_name: str = "JGD2011") -> pyproj.CRS:
    """
    ## Summary:
        経緯度（度単位）からUTM座標系を推定する。
    Args:
        lon (float): 経度
        lat (float): 緯度
        datum_name(str): 'WGS 84', 'JGD2011' ...  default='JGD2011'
    Returns:
        pyproj.CRS: 推定されたUTM座標系のCRSオブジェクト
    """
    try:
        # Check if the datum_name is valid
        pyproj.CRS.from_user_input(datum_name)
    except Exception as e:
        raise ValueError("Invalid datum_name. Use 'WGS 84', 'JGD2011', etc.") from e
    # Estimate the UTM CRS
    aoi = pyproj.aoi.AreaOfInterest(
        west_lon_degree=lon,
        south_lat_degree=lat,
        east_lon_degree=lon,
        north_lat_degree=lat,
    )
    utm_crs_lst = pyproj.database.query_utm_crs_info(
        datum_name=datum_name, area_of_interest=aoi
    )
    return pyproj.CRS.from_epsg(utm_crs_lst[0].code)


@type_checker_shapely(arg_index=0, kward="geometry")
@type_checker_crs(arg_index=1, kward="in_crs")
def estimate_utm_crs_from_geometry(
    geometry: BaseGeometry,
    in_crs: str | int | pyproj.CRS = "EPSG:4326",
    datum_name: str = "JGD2011",
) -> pyproj.CRS:
    """
    ## Summary:
        ジオメトリからUTM座標系を推定する。
    Args:
        geometry (BaseGeometry):
            ジオメトリ。shapelyのBaseGeometryオブジェクト。
        in_crs (str | int | pyproj.CRS):
            入力座標系。EPSGコードやCRSオブジェクトを指定。デフォルトは'EPSG:4326'。
        datum_name(str): 'WGS 84', 'JGD2011' ...  default='JGD2011'
    Returns:
        pyproj.CRS: 推定されたUTM座標系のCRSオブジェクト
    """
    if in_crs.to_epsg() != 4326:
        # 入力座標系が経緯度でない場合、変換を行う
        geometry = transform_geometry(geometry, in_crs, "EPSG:4326")
    pnt = geometry.centroid
    return estimate_utm_crs(pnt.x, pnt.y, datum_name)


@type_checker_crs(arg_index=1, kward="in_crs")
@type_checker_crs(arg_index=2, kward="out_crs")
def get_geometry_center(
    geometry: BaseGeometry | list[BaseGeometry],
    in_crs: str | int | pyproj.CRS,
    out_crs: str | int | pyproj.CRS,
) -> XY:
    # ``geometry``の次元数を数えて、問題がなければBBoxの中心を取得する。
    dim_count = dimensional_count(geometry)
    if dim_count == 1:
        geometry = shapely.union_all(geometry).envelope.centroid
    elif dim_count == 0:
        geometry = geometry.envelope.centroid
    else:
        raise ValueError("geometry must be a single geometry or a list of geometries.")
    # ``geometry``のCRSが`in_crs`と異なる場合、変換を行う。
    if in_crs.to_epsg() != out_crs.to_epsg():
        geometry = transform_geometry(geometry, in_crs, out_crs)
    return XY(geometry.x, geometry.y)


@type_checker_crs(arg_index=1, kward="in_crs")
@type_checker_crs(arg_index=2, kward="out_crs")
def get_geometry_scope(
    geometry: BaseGeometry | list[BaseGeometry],
    in_crs: str | int | pyproj.CRS,
    out_crs: str | int | pyproj.CRS,
) -> Scope:
    """
    ## Summary:
        ジオメトリの範囲（バウンディングボックス）を取得する。
    Args:
        geometry (BaseGeometry | list[BaseGeometry]):
            ジオメトリ。shapelyのBaseGeometryオブジェクトまたはそのリスト。
        in_crs (str | int | pyproj.CRS):
            入力座標系。EPSGコードやCRSオブジェクトを指定。
        out_crs (str | int | pyproj.CRS):
            出力座標系。EPSGコードやCRSオブジェクトを指定。
    Returns:
        Scope(NamedTuple):
            ジオメトリの範囲を表すScopeオブジェクト。
            - x_min: float
            - y_min: float
            - x_max: float
            - y_max: float
    """
    dim_count = dimensional_count(geometry)
    if dim_count == 1:
        geometry = shapely.union_all(geometry).envelope
    elif dim_count == 0:
        geometry = geometry.envelope
    else:
        raise ValueError("geometry must be a single geometry or a list of geometries.")
    # ``geometry``のCRSが`in_crs`と異なる場合、変換を行う。
    if in_crs.to_epsg() != out_crs.to_epsg():
        geometry = transform_geometry(geometry, in_crs, out_crs)
    return Scope(*geometry.bounds)


def get_coordinates_from(
    pnt: shapely.Point, degree: float, distance: float
) -> shapely.Point:
    """
    ## Summary:
        任意の地点から、方向と距離を指定した位置を求める
    Args:
        pnt(shapely.Point):
            開始位置
        degree(float):
            方向角
        distance(float):
            距離
    Returns:
        shapely.Point:
    """
    from decimal import Decimal

    tmp = (Decimal("90.0") - Decimal(f"{degree}")) % Decimal("360.0")
    radians = math.radians(float(tmp))
    x = pnt.x + math.cos(radians) * distance
    y = pnt.y + math.sin(radians) * distance
    return shapely.Point(x, y)
