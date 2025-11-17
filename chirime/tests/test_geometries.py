from decimal import Decimal

import pyproj
import pytest
import shapely

from chiriin.config import XY, Scope
from chiriin.geometries import (
    degree_to_dms,
    degree_to_dms_lonlat,
    dms_to_degree,
    dms_to_degree_lonlat,
    estimate_utm_crs,
    estimate_utm_crs_from_geometry,
    get_geometry_center,
    get_geometry_scope,
    transform_geometry,
    transform_xy,
)

DMS1_LON = 1403906.4277
DEG1_LON = 140.651785472
DMS1_LAT = 405109.1161
DEG1_LAT = 40.85253225


@pytest.mark.parametrize(
    "dms, digits, decimal_obj, expected",
    [
        (DMS1_LON, 9, False, DEG1_LON),
        (DMS1_LON, 9, True, Decimal(f"{DEG1_LON}")),
        (DMS1_LAT, 9, False, DEG1_LAT),
    ],
)
def test_dms_to_degree(dms, digits, decimal_obj, expected):
    """Test the conversion from DMS to decimal degrees."""
    result = dms_to_degree(dms, digits, decimal_obj)
    if decimal_obj:
        assert isinstance(result, Decimal)
    else:
        assert isinstance(result, float)
        assert result == pytest.approx(expected, rel=1 * 10**-digits)
    with pytest.raises(ValueError):
        dms_to_degree("invalid_dms", digits, decimal_obj)  # type: ignore
    with pytest.raises(ValueError):
        dms_to_degree(1400000000.1234, 9, False)


@pytest.mark.parametrize(
    "lon, lat, digits, decimal_obj, expected",
    [
        (DMS1_LON, DMS1_LAT, 9, False, XY(x=DEG1_LON, y=DEG1_LAT)),
        (
            DMS1_LON,
            DMS1_LAT,
            9,
            True,
            XY(x=Decimal(f"{DEG1_LON}"), y=Decimal(f"{DEG1_LAT}")),
        ),
        ([DMS1_LON], [DMS1_LAT], 9, False, [XY(x=DEG1_LON, y=DEG1_LAT)]),
    ],
)
def test_dms_to_lonlat(lon, lat, digits, decimal_obj, expected):
    """Test the conversion from DMS to lonlat."""
    result = dms_to_degree_lonlat(lon, lat, digits, decimal_obj)
    if isinstance(expected, list):
        for res_xy, exp_xy in zip(result, expected, strict=False):
            assert isinstance(res_xy, XY)
            if not decimal_obj:
                assert res_xy.x == pytest.approx(exp_xy.x, rel=1 * 10**-digits)
                assert res_xy.y == pytest.approx(exp_xy.y, rel=1 * 10**-digits)
    else:
        assert isinstance(result, XY)
        if not decimal_obj:
            assert result.x == pytest.approx(expected.x, rel=1 * 10**-digits)
            assert result.y == pytest.approx(expected.y, rel=1 * 10**-digits)


@pytest.mark.parametrize(
    "degree, digits, decimal_obj, expected",
    [
        (DEG1_LON, 4, False, DMS1_LON),
        (DEG1_LON, 4, True, Decimal(f"{DMS1_LON}")),
        (DEG1_LAT, 4, False, DMS1_LAT),
    ],
)
def test_degree_to_dms(degree, digits, decimal_obj, expected):
    """Test the conversion from decimal degrees to DMS."""
    result = degree_to_dms(degree, digits, decimal_obj)
    if decimal_obj:
        assert isinstance(result, Decimal)
    else:
        assert isinstance(result, float)
        assert result == pytest.approx(expected, rel=1 * 10**-digits)

    with pytest.raises(ValueError):
        degree_to_dms("invalid_degree", digits, decimal_obj)  # type: ignore
    with pytest.raises(ValueError):
        degree_to_dms(200, 10, False)


@pytest.mark.parametrize(
    "lon, lat, digits, decimal_obj, expected",
    [
        (DEG1_LON, DEG1_LAT, 4, False, (DMS1_LON, DMS1_LAT)),
        (DEG1_LON, DEG1_LAT, 4, True, (Decimal(f"{DMS1_LON}"), Decimal(f"{DMS1_LAT}"))),
        ([DEG1_LON], [DEG1_LAT], 4, False, [(DMS1_LON, DMS1_LAT)]),
    ],
)
def test_degree_to_lonlat(lon, lat, digits, decimal_obj, expected):
    """Test the conversion from lonlat to DMS."""
    result = degree_to_dms_lonlat(lon, lat, digits, decimal_obj)
    if isinstance(expected, list):
        assert isinstance(result, list)
        for res_xy, exp_xy in zip(result, expected, strict=False):
            assert isinstance(res_xy, XY)
            if not decimal_obj:
                assert res_xy.x == pytest.approx(exp_xy[0], rel=1 * 10**-digits)
                assert res_xy.y == pytest.approx(exp_xy[1], rel=1 * 10**-digits)
    else:
        assert isinstance(result, XY)
        if not decimal_obj:
            assert result.x == pytest.approx(expected[0], rel=1 * 10**-digits)
            assert result.y == pytest.approx(expected[1], rel=1 * 10**-digits)


def test_transform_xy():
    """Test the transformation of coordinates between different CRS."""
    lon, lat = 140.651785472, 40.85253225
    in_crs = "EPSG:4326"
    out_crs = "EPSG:3857"

    transformed_xy = transform_xy(lon, lat, in_crs, out_crs)
    assert isinstance(transformed_xy, XY)
    assert transformed_xy.x != lon or transformed_xy.y != lat, (
        "Coordinates should be transformed."
    )

    # Test with invalid CRS
    with pytest.raises(Exception):  # noqa: B017
        transform_xy(lon, lat, "invalid_crs", out_crs)


@pytest.mark.parametrize(
    "geometry, in_crs, out_crs",
    [
        (shapely.Point(140.651785472, 40.85253225), "EPSG:4326", "EPSG:3857"),
        (
            shapely.Point(140.651785472, 40.85253225).buffer(0.1).envelope,
            "EPSG:4326",
            "EPSG:6678",
        ),
    ],
)
def test_transform_geometry(geometry, in_crs, out_crs):
    """Test the transformation of geometries between coordinate reference systems."""
    transformed_geometry = transform_geometry(geometry, in_crs, out_crs)
    assert isinstance(transformed_geometry, shapely.geometry.base.BaseGeometry)
    assert geometry != transformed_geometry, "Geometry should be transformed."
    # Test with invalid CRS
    with pytest.raises(Exception):  # noqa: B017
        transform_geometry(geometry, "invalid_crs", out_crs)


def test_estimate_utm_crs():
    """Test the estimation of UTM CRS from coordinates."""
    lon, lat = 140.651785472, 40.85253225
    utm_crs = estimate_utm_crs(lon, lat)
    assert isinstance(utm_crs, pyproj.CRS)
    assert utm_crs.axis_info[0].unit_name == "metre", (
        "Estimated CRS should use metres as the unit."
    )
    with pytest.raises(Exception):  # noqa: B017
        estimate_utm_crs(lon, lat, datum_name="invalid_datum")


def test_estimate_utm_crs_from_geometry():
    """Test the estimation of UTM CRS from a geometry."""
    geometry = shapely.Point(140.651785472, 40.85253225)
    utm_crs = estimate_utm_crs_from_geometry(geometry, in_crs="EPSG:4326")
    assert isinstance(utm_crs, pyproj.CRS)
    assert utm_crs.axis_info[0].unit_name == "metre", (
        "Estimated CRS should use metres as the unit."
    )


def test_get_geometry_center():
    """Test the calculation of the center of a geometry."""
    geometry = shapely.Point(140.651785472, 40.85253225).buffer(0.1).envelope
    center = get_geometry_center(geometry, in_crs="EPSG:4326", out_crs="EPSG:3857")
    assert isinstance(center, XY)
    center = get_geometry_center([geometry], in_crs="EPSG:4326", out_crs="EPSG:3857")
    assert isinstance(center, XY)
    # Test with invalid geometry
    with pytest.raises(Exception):  # noqa: B017
        get_geometry_center([[geometry]], in_crs="EPSG:4326", out_crs="EPSG:3857")


def test_get_geometry_scope():
    """Test the calculation of the scope of a geometry."""
    geometry = shapely.Point(140.651785472, 40.85253225).buffer(0.1).envelope
    scope = get_geometry_scope(geometry, in_crs="EPSG:4326", out_crs="EPSG:3857")
    assert isinstance(scope, Scope)
    scope = get_geometry_scope([geometry], in_crs="EPSG:4326", out_crs="EPSG:3857")
    assert isinstance(scope, Scope)
    # Test with invalid geometry
    with pytest.raises(Exception):  # noqa: B017
        get_geometry_scope([[geometry]], in_crs="EPSG:4326", out_crs="EPSG:3857")
