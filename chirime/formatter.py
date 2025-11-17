import datetime
from decimal import Decimal
from typing import Any, Iterable

import pyproj
import shapely

from .utils import dimensional_count

DEM_TYPES = ["dem10b", "dem5a", "dem5b"]
IMG_TYPES = ["standard", "photo", "slope", "google_satellite"]


def datetime_formatter(dt: datetime.datetime | str) -> datetime.datetime:
    """
    ## Description:
        日時のフォーマットを統一する関数
        datetimeオブジェクトまたは文字列を受け取り、マイクロ秒を0にして返す
    ## Args:
        dt (datetime.datetime | str):
            日時を表すdatetimeオブジェクトまたは文字列
    ## Returns:
        datetime.datetime:
            マイクロ秒が0に設定されたdatetimeオブジェクト
    """
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%f+%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]

    if isinstance(dt, datetime.datetime):
        return dt.replace(microsecond=0)

    elif isinstance(dt, str):
        for fmt in fmts:
            try:
                return datetime.datetime.strptime(dt, fmt).replace(microsecond=0)
            except ValueError:
                continue
        try:
            return datetime.datetime.fromisoformat(dt).replace(tzinfo=None, microsecond=0)
        except ValueError:
            raise ValueError(f"Unsupported datetime format: {dt}")  # noqa: B904

    raise TypeError(f"Expected datetime or str, got {type(dt)}")


def _intermediate(arg_index, kward, *args, **kwargs) -> dict[str, Any]:
    """
    ## Description:
        引数が args にあるか kwargs にあるかを判定するヘルパー関数。
    ## Args:
        arg_index (int):
            位置引数のインデックス。
        kward (str):
            キーワード引数の名前。
        *args:
            可変長引数リスト。
        **kwargs:
            任意のキーワード引数。
    ## Returns:
        dict:
            辞書型で、引数が args にあるかどうかとその値を含む。
            "in_args" (bool): 引数が args にある場合は True、kwargs にある場合は False。
            "value" (Any): 引数の値。
    """
    in_args = True
    value = None
    if arg_index < len(args):
        value = args[arg_index]
    else:
        in_args = False
        value = kwargs[kward]
    return {"in_args": in_args, "value": value}


def _return_value(value: Any, data: dict[str, Any], args, kwargs) -> Any:
    """
    ## Description:
        Helper function to return the modified args and kwargs after type checking.
    ## Args:
        value (Any):
            The value to be set in args or kwargs.
        data (dict[str, Any]):
            The data containing information about the argument index and keyword.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.
    ## Returns:
        dict:
            A dictionary containing the modified args and kwargs.
    """
    if data["in_args"]:
        args = list(args)
        args[data["arg_index"]] = value
    else:
        kwargs[data["kward"]] = value
    return {"args": args, "kwargs": kwargs}


def type_checker_float(arg_index: int, kward: str):
    """
    ## Description:
        引数が浮動小数点数か浮動小数点数に変換可能かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        float:
            浮動小数点数に変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            try:
                value = float(value)
            except Exception as e:
                raise TypeError(
                    f"Argument '{kward}' must be a float or convertible to float"
                    f", got {type(value)}"
                ) from e
            else:
                result = _return_value(value, data, args, kwargs)
                return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_integer(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数が整数か、整数に変換可能かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        int:
            整数に変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            try:
                value = int(value)
            except Exception as e:
                raise TypeError(
                    f"Argument '{kward}' must be an integer or convertible to "
                    f"integer, got {type(value)}"
                ) from e
            else:
                result = _return_value(value, data, args, kwargs)
                return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_datetime(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数がdatetimeオブジェクトか、datetimeに変換可能な文字列かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        datetime.datetime:
            datetimeオブジェクトに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            value = datetime_formatter(value)
            result = _return_value(value, data, args, kwargs)
            return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_decimal(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数がDecimalオブジェクトか、Decimalに変換可能な値かをチェックするデコレーター。
        Decimalは浮動小数点数の精度を保つために使用されます。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        float:
            Decimalに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            if isinstance(value, Decimal):
                return func(*args, **kwargs)
            try:
                value = Decimal(f"{float(value)}")
            except Exception as e:
                raise TypeError(
                    f"Argument '{kward}' must be a decimal or convertible to "
                    "decimal, got {type(value)}"
                ) from e
            else:
                result = _return_value(value, data, args, kwargs)
                return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_iterable(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数が一次元の繰り返し可能なオブジェクトかをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        Iterable:
            一次元の繰り返し可能なオブジェクト。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            count = dimensional_count(value)
            if count == 0:
                value = [value]
            elif count != 1:
                raise TypeError(
                    f"Argument '{kward}' must be a one-dimensional iterable, "
                    f"got {count}D iterable."
                )
            result = _return_value(value, data, args, kwargs)
            return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_crs(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数がpyproj.CRSオブジェクトか、CRSに変換可能な文字列かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        pyproj.CRS:
            CRSオブジェクトに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            if isinstance(value, pyproj.CRS):
                return func(*args, **kwargs)
            try:
                if isinstance(value, str):
                    value = pyproj.CRS(value)
                else:
                    value = pyproj.CRS.from_epsg(value)
            except Exception as e:
                raise TypeError(
                    f"Argument '{kward}' must be a CRS or convertible to CRS, got {type(value)}"
                ) from e
            else:
                result = _return_value(value, data, args, kwargs)
                return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_shapely(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数がshapelyオブジェクトか、shapelyに変換可能な値かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        shapely.geometry.base.BaseGeometry:
            shapelyオブジェクトに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            if shapely.is_geometry(value):
                return func(*args, **kwargs)
            try:
                value = shapely.from_wkt(value)
            except Exception as e:
                raise TypeError(
                    f"Argument '{kward}' must be a shapely object or convertible to shapely, got {type(value)}"
                ) from e
            else:
                result = _return_value(value, data, args, kwargs)
                return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_zoom_level(
    arg_index: int, kward: str, min_zl: int = 0, max_zl: int = 24
):
    """
    ## Description:
        関数の引数がズームレベルを表す整数か、整数に変換可能かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
        min_zl (int):
            ズームレベルの最小値。デフォルトは0。
        max_zl (int):
            ズームレベルの最大値。デフォルトは24。
    ## Returns:
        int:
            ズームレベルに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            try:
                value = int(value)
                if not (min_zl <= value <= max_zl):
                    raise ValueError(
                        f"Zoom level must be between {min_zl} and {max_zl}, got {value}"
                    )
            except Exception as e:
                raise TypeError(
                    f"Argument '{kward}' must be an integer or convertible to integer, got {type(value)}"
                ) from e
            else:
                result = _return_value(value, data, args, kwargs)
                return func(*result["args"], **result["kwargs"])

        return wrapper

    return decorator


def type_checker_elev_type(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数が標高タイプを表す文字列か、整数に変換可能かをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        str:
            標高タイプに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            if isinstance(value, str):
                value = value.lower()
                if value in DEM_TYPES:
                    result = _return_value(value, data, args, kwargs)
                    return func(*result["args"], **result["kwargs"])
                else:
                    raise ValueError(
                        f"Invalid elevation type '{value}'. Must be 'dem10b', "
                        "'dem5a', or 'dem5b'."
                    )
            else:
                raise TypeError(
                    f"Argument '{kward}' must be a string representing elevation "
                    f"type, got {type(value)}"
                )

        return wrapper

    return decorator


def type_checker_img_type(arg_index: int, kward: str):
    """
    ## Description:
        関数の引数が地理院のタイルの種類を特定可能かどうかをチェックするデコレーター。
    ## Args:
        arg_index (int):
            位置引数のインデックスを指定。
        kward (str):
            キーワード引数の名前を指定。
    ## Returns:
        str:
            画像タイプに変換された引数の値。
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = _intermediate(arg_index, kward, *args, **kwargs)
            data["arg_index"] = arg_index
            data["kward"] = kward
            value = data["value"]
            if isinstance(value, str):
                value = value.lower()
                if value in IMG_TYPES:
                    result = _return_value(value, data, args, kwargs)
                    return func(*result["args"], **result["kwargs"])
                else:
                    raise ValueError(
                        f"Invalid image type '{value}'. Must be one of {IMG_TYPES}."
                    )
            else:
                raise TypeError(
                    f"Argument '{kward}' must be a string representing image "
                    f"type, got {type(value)}"
                )

        return wrapper

    return decorator


@type_checker_float(arg_index=0, kward="value")
def float_formatter(value: int | float | str) -> float:
    """
    ## Description:
        浮動小数点数のフォーマットを統一する関数。
    ## Args:
        value (int | float | str):
            浮動小数点に変換したい値。
    ## Returns:
        float:
            フォーマットされた浮動小数点数。
    """
    return value  # type: ignore


@type_checker_integer(arg_index=0, kward="value")
def integer_formatter(value: int | float | str) -> int:
    """
    ## Description:
        整数のフォーマットを統一する関数。
        引数が整数、浮動小数点数、または整数に変換可能な文字列であることを確認します。
    ## Args:
        value (int | float | str):
            整数に変換したい値。
            整数、浮動小数点数、または整数に変換可能な文字列を受け入れます。
    ## Returns:
        int:
            フォーマットされた整数。
    """
    return value  # type: ignore


def iterable_float_formatter(values: Iterable) -> list[float]:
    """
    ## Description:
        繰り返し処理可能なオブジェクト内の値を浮動小数点数のリストに変換する関数。
    ## Args:
        values (Iterable):
            一次元の繰り返し可能なオブジェクト。
            リスト、タプル、または数値を含む他の繰り返し可能なオブジェクトを受け入れます。
    ## Returns:
        list[float]:
            フォーマットされた浮動小数点数のリスト。
    """
    count = dimensional_count(values)
    assert count == 1, f"Expected one-dimensional iterable, got {count}D iterable."
    return [float_formatter(value) for value in values]


def iterable_integer_formatter(values: Iterable) -> list[int]:
    """
    ## Description:
        繰り返し可能なオブジェクト内の値を整数のリストに変換する関数。
    ## Args:
        values (Iterable):
            一次元の繰り返し可能なオブジェクト。
            リスト、タプル、または数値を含む他の繰り返し可能なオブジェクトを受け入れます。
    ## Returns:
        list[int]:
            フォーマットされた整数のリスト。
    """
    count = dimensional_count(values)
    assert count == 1, f"Expected one-dimensional iterable, got {count}D iterable."
    return [integer_formatter(value) for value in values]


def iterable_decimalize_formatter(values: Iterable) -> list[Decimal]:
    """
    ## Description:
        繰り返し可能なオブジェクト内の値をDecimalのリストに変換する関数。
    ## Args:
        values (Iterable):
            一次元の繰り返し可能なオブジェクト。
            リスト、タプル、または数値を含む他の繰り返し可能なオブジェクトを受け入れます。
    ## Returns:
        list[Decimal]:
            フォーマットされたDecimalのリスト。
    """
    count = dimensional_count(values)
    assert count == 1, f"Expected one-dimensional iterable, got {count}D iterable."
    return [Decimal(f"{float(value)}") for value in values]


def crs_formatter(crs: str | int | pyproj.CRS) -> pyproj.CRS:
    """
    ## Description:
        CRS（座標参照系）をフォーマットする関数。
        引数がCRSオブジェクト、文字列、または整数であることを確認します。
    ## Args:
        crs (str | int | pyproj.CRS):
            CRSを表す文字列、整数、またはpyproj.CRSオブジェクト。
    ## Returns:
        pyproj.CRS:
            フォーマットされたCRSオブジェクト。
    """
    if isinstance(crs, pyproj.CRS):
        return crs
    elif isinstance(crs, str):
        return pyproj.CRS(crs)
    elif isinstance(crs, int):
        return pyproj.CRS.from_epsg(crs)
    else:
        raise TypeError(
            f"Expected CRS to be a string, integer, or pyproj.CRS object, got {type(crs)}"
        )
