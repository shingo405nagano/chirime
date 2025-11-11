from decimal import Decimal
from typing import Iterable

import pyproj

from .data import XY
from .formatter import type_checker_crs, type_checker_float


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


def dms_to_degree(
    dms: float,  #
    digits: int = 9,
    decimal_obj: bool = False,
) -> float | Decimal:
    """
    ## Summary:
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


def dms_to_degree_lonlat(
    lon: float,  #
    lat: float,
    digits: int = 9,
    decimal_obj: bool = False,
) -> XY:
    """
    ## Summary:
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
