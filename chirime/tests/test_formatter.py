import datetime
from decimal import Decimal

import pyproj
import pytest
import shapely

from chiriin.formatter import (
    datetime_formatter,
    float_formatter,
    integer_formatter,
    iterable_decimalize_formatter,
    iterable_float_formatter,
    iterable_integer_formatter,
    type_checker_crs,
    type_checker_datetime,
    type_checker_decimal,
    type_checker_elev_type,
    type_checker_float,
    type_checker_img_type,
    type_checker_integer,
    type_checker_iterable,
    type_checker_shapely,
    type_checker_zoom_level,
)


@pytest.mark.parametrize(
    "datetime_",
    [
        "2023-11-16T11:06:21.700+09:00",
        "2023-11-16T11:06:21.700",
        "2023-11-16 11:06:21",
        "2023-11-16",
        "2023/11/16 11:06:21",
        "2023/11/16 11:06",
        datetime.datetime(2023, 11, 16, 11, 6, 21, 700000),
    ],
)
def test_datetime_formatter(datetime_):
    """Test datetime_formatter function."""
    result = datetime_formatter(datetime_)
    assert isinstance(result, datetime.datetime)
    with pytest.raises(ValueError):
        datetime_formatter("invalid datetime string")
    with pytest.raises(TypeError):
        datetime_formatter(12345)  # type: ignore


@pytest.mark.parametrize(
    "value, expected, success",
    [
        (1, 1.0, True),
        (1.0, 1.0, True),
        (1.5, 1.5, True),
        ("1", 1.0, True),
        ("10-", None, False),
    ],
)
def test_type_checker_float(value, expected, success):
    @type_checker_float(arg_index=0, kward="value")
    def dummy_function(value: float):
        return value

    if success:
        result = dummy_function(value)
        # result = dummy_function(value=value)
        assert isinstance(result, float)
        assert result == expected
    else:
        with pytest.raises(TypeError):
            dummy_function(value)


@pytest.mark.parametrize(
    "value, expected, success",
    [
        (1, 1, True),
        (1.0, 1, True),
        ("1", 1, True),
        ("10-", None, False),
    ],
)
def test_type_checker_integer(value, expected, success):
    @type_checker_integer(arg_index=0, kward="value")
    def dummy_function(value: int):
        return value

    if success:
        result = dummy_function(value)
        # result = dummy_function(value=value)
        assert isinstance(result, int)
        assert result == expected
    else:
        with pytest.raises(TypeError):
            dummy_function(value)


@pytest.mark.parametrize(
    "datetime_, success",
    [
        ("2023-11-16T11:06:21.700+09:00", True),
        ("2023-11-16T11:06:21.700", True),
        ("2023-11-16 11:06:21", True),
        ("2023-11-16", True),
        ("2023/11/16 11:06:21", True),
        ("2023/11/16 11:06", True),
        (datetime.datetime(2023, 11, 16, 11, 6, 21, 700000), True),
        ("invalid datetime string", False),
        (12345, False),  # type: ignore
    ],
)
def test_type_checker_datetime(datetime_, success):
    @type_checker_datetime(arg_index=0, kward="datetime_")
    def dummy_function(datetime_: datetime.datetime):
        return datetime_

    if success:
        result = dummy_function(datetime_)
        result = dummy_function(datetime_=datetime_)
        assert isinstance(result, datetime.datetime)
        assert result.microsecond == 0
    else:
        with pytest.raises((ValueError, TypeError)):
            dummy_function(datetime_)


@pytest.mark.parametrize(
    "value, success",
    [
        (1, True),
        (1.0, True),
        ("1", True),
        ("10-", False),
    ],
)
def test_type_checker_decimal(value, success):
    @type_checker_decimal(arg_index=0, kward="value")
    def dummy_function(value: float):
        return value

    if success:
        result = dummy_function(value)
        assert isinstance(result, Decimal)
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(value)


@pytest.mark.parametrize(
    "value, success",
    [
        (1, True),
        ([1, 2, 3], True),
        ([[1, 2], [3, 4]], False),
    ],
)
def test_type_checker_iterable(value, success):
    @type_checker_iterable(arg_index=0, kward="value")
    def dummy_function(value: list):
        return value

    if success:
        result = dummy_function(value)
        assert isinstance(result, list)
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(value)


@pytest.mark.parametrize("value", [100, 100.0, "100"])
def test_float_formatter(value):
    """Test float_formatter function."""
    result = float_formatter(value)
    assert isinstance(result, float)


@pytest.mark.parametrize("value", [100, 100.0, "100"])
def test_integer_formatter(value):
    """Test integer_formatter function."""
    result = integer_formatter(value)
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "value, success",
    [
        (100, False),
        ([1, 2, 3], True),
        ([1.0, 2.0, 3.0], True),
        (["1", "2", "3"], True),
        (["1.0", "2.0", "3.0"], True),
        ("100", False),
        ([[1, 2], [3, 4]], False),
    ],
)
def test_iterable_float_formatter(value, success):
    """Test iterable_float_formatter function."""
    if success:
        result = iterable_float_formatter(value)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
    else:
        with pytest.raises(Exception):  # noqa: B017
            iterable_float_formatter(value)


@pytest.mark.parametrize(
    "value, success",
    [
        (100, False),
        ([1, 2, 3], True),
        ([1.0, 2.0, 3.0], True),
        (["1", "2", "3"], True),
        ("100", False),
        ([[1, 2], [3, 4]], False),
    ],
)
def test_iterable_integer_formatter(value, success):
    """Test iterable_integer_formatter function."""
    if success:
        result = iterable_integer_formatter(value)
        assert isinstance(result, list)
        assert all(isinstance(v, int) for v in result)
    else:
        with pytest.raises(Exception):  # noqa: B017
            iterable_integer_formatter(value)


@pytest.mark.parametrize(
    "crs, success",
    [
        ("EPSG:4326", True),
        ("EPSG:3857", True),
        ("invalid_crs", False),
        (6678, True),
        (pyproj.CRS.from_epsg(4326), True),
    ],
)
def test_type_checker_crs(crs, success):
    """Test type_checker_crs function."""

    @type_checker_crs(arg_index=0, kward="crs")
    def dummy_function(crs: pyproj.CRS):
        return crs

    if success:
        result = dummy_function(crs)
        assert isinstance(result, pyproj.CRS)
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(crs)


@pytest.mark.parametrize(
    "value, success",
    [
        (shapely.geometry.Point(1, 2), True),
        (shapely.geometry.LineString([(0, 0), (1, 1)]), True),
        (shapely.geometry.Polygon([(0, 0), (1, 1), (1, 0)]), True),
        ("invalid_shape", False),
        ("POINT (0 0)", True),
    ],
)
def test_type_checker_shapely(value, success):
    """Test type_checker_shapely function."""

    @type_checker_shapely(arg_index=0, kward="value")
    def dummy_function(value):
        return value

    if success:
        result = dummy_function(value)
        assert shapely.is_geometry(result)
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(value)


def test_iterable_decimalize_formatter():
    """Test iterable_decimalize_formatter function."""
    values = [1, 2.5, "3.14", "4"]
    result = iterable_decimalize_formatter(values)
    assert isinstance(result, list)
    assert all(isinstance(v, Decimal) for v in result)


@pytest.mark.parametrize(
    "zl, min_zl, max_zl, success",
    [
        (10, 0, 24, True),
        (100, 0, 24, False),
        ("1", 0, 24, True),
        ("invalid", 0, 24, False),
    ],
)
def test_type_checker_zoom_level(zl, min_zl, max_zl, success):
    """Test type_checker_zoom_level function."""

    @type_checker_zoom_level(arg_index=0, kward="zl", min_zl=min_zl, max_zl=max_zl)
    def dummy_function(zl: int):
        return zl

    if success:
        result = dummy_function(zl)
        assert isinstance(result, int)
        assert min_zl <= result <= max_zl
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(zl)


@pytest.mark.parametrize(
    "value, expected, success",
    [
        ("dem10b", "dem10b", True),
        ("dem5a", "dem5a", True),
        ("dem5b", "dem5b", True),
        ("invalid", None, False),
        (100, None, False),
    ],
)
def test_type_checker_elev_type(value, expected, success):
    """Test type_checker_elev_type function."""

    @type_checker_elev_type(arg_index=0, kward="value")
    def dummy_function(value: str):
        return value

    if success:
        result = dummy_function(value)
        assert isinstance(result, str)
        assert result == expected
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(value)


@pytest.mark.parametrize(
    "value, expected, success",
    [
        ("standard", "standard", True),
        ("STANDARD", "standard", True),
        ("photo", "photo", True),
        ("slope", "slope", True),
        ("invalid", None, False),
    ],
)
def test_type_checker_img_type(value, expected, success):
    """Test type_checker_img_type function."""

    @type_checker_img_type(arg_index=0, kward="value")
    def dummy_function(value: str):
        return value

    if success:
        result = dummy_function(value)
        assert isinstance(result, str)
        assert result == expected
    else:
        with pytest.raises(Exception):  # noqa: B017
            dummy_function(value)
