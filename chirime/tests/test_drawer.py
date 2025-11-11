import math

import numpy as np
import pytest
import shapely

from chiriin.config import XY, XYZ, RelativePosition, TileData, TileScope
from chiriin.geometries import transform_xy
from chiriin.mesh import MeshCode
from _drawer import chiriin_drawer


@pytest.mark.parametrize(
    "x, y, success",
    [
        (140.087099, 40.504665, True),
        (139.087099, 35.504665, True),
        ("140.087099", "40.504665", True),
        (140.087099, -40.504665, False),
    ],
)
def test_magnetic_declination_from_chiriin_drawer(x, y, success):
    """Test the magnetic_declination function."""
    if success:
        res = chiriin_drawer.magnetic_declination(x, y)
        assert isinstance(res, float)
        assert -180 <= res <= 180
    else:
        assert math.isnan(chiriin_drawer.magnetic_declination(x, y))


@pytest.mark.parametrize(
    "x, y, success",
    [
        (140.087099, 40.504665, True),
        (139.087099, 35.504665, True),
        (140.087099, "40.504665", True),
        (140.087099, "invalid", False),
    ],
)
def test_get_chmesh_code_from_chiriin_drawer(x, y, success):
    """Test the get_mesh_code function."""
    if success:
        res = chiriin_drawer.get_mesh_code(x, y)
        assert isinstance(res, MeshCode)
    else:
        with pytest.raises(Exception):  # noqa: B017
            chiriin_drawer.get_mesh_code(x, y)


def test_semidynamic_2d_from_chiriin_drawer():
    """Test the semidynamic_2d function."""
    x = 140.769399
    y = 39.769496
    datetimes = [
        "2020-10-01 00:00:00",
        "2021-10-01 12:00:00",
        "2022-10-02 00:00:00",
        "2023-10-02 12:00:00",
    ]
    result_x = []
    result_y = []
    for dt in datetimes:
        res = chiriin_drawer.semidynamic_2d(
            lon=x,
            lat=y,
            datetime_=dt,
        )
        assert isinstance(res, XY)
        assert res.x != x
        assert res.y != y
        assert res.x not in result_x
        assert res.y not in result_y
        result_x.append(res.x)
        result_y.append(res.y)


def test_fetch_semidynamic_2d_from_chiriin_drawer():
    """Test the semidynamic_2d_with_web_api function."""
    x = 140.769399
    y = 39.769496
    datetimes = [
        "2020-10-01 00:00:00",
        "2021-10-01 12:00:00",
        "2022-10-02 00:00:00",
        "2023-10-02 12:00:00",
    ]
    result_x = []
    result_y = []
    for dt in datetimes:
        res = chiriin_drawer.fetch_semidynamic_2d(
            lon=x,
            lat=y,
            datetime_=dt,
        )
        assert isinstance(res, XY)
        assert res.x != x
        assert res.y != y
        assert res.x not in result_x
        assert res.y not in result_y
        result_x.append(res.x)
        result_y.append(res.y)


def test_fetch_semidynamic_3d_from_chiriin_drawer():
    """Test the semidynamic_3d_with_web_api function."""
    x = 140.769399
    y = 39.769496
    z = 10
    datetimes = [
        "2020-10-01 00:00:00",
        "2021-10-01 12:00:00",
        "2022-10-02 00:00:00",
        "2023-10-02 12:00:00",
    ]
    result_x = []
    result_y = []
    for dt in datetimes:
        res = chiriin_drawer.fetch_semidynamic_3d(
            lon=x,
            lat=y,
            altitude=z,
            datetime_=dt,
        )
        assert isinstance(res, XYZ)
        assert res.x != x
        assert res.y != y
        assert res.x not in result_x
        assert res.y not in result_y
        result_x.append(res.x)
        result_y.append(res.y)


def test_fetch_distance_and_azimuth_from_chiriin_drawer():
    """Test the distance_and_azimuth_with_web_api function."""
    x1 = 140.769399
    y1 = 39.769496
    x2 = 140.769500
    y2 = 39.769600
    res = chiriin_drawer.fetch_distance_and_azimuth(
        lon1=x1,
        lat1=y1,
        lon2=x2,
        lat2=y2,
    )
    assert isinstance(res, RelativePosition)


def test_fetch_elevation_from_chiriin_drawer():
    """Test the fetch_elevation function."""
    x = 140.769399
    y = 39.769496
    res = chiriin_drawer.fetch_elevation(x, y, in_crs="EPSG:4326")
    assert isinstance(res, float)
    xs = [140.769399, 140.769500]
    ys = [39.769496, 39.769600]
    res = chiriin_drawer.fetch_elevation(xs, ys, in_crs="EPSG:4326")
    assert isinstance(res, list)
    assert all([isinstance(r, float) for r in res])


def test_fetch_elevation_tile_xy_from_chiriin_drawer():
    """Test the get_elevation_tile_xy function."""
    x = 140.769399
    y = 39.769496
    zl = 14
    resps = chiriin_drawer.fetch_elevation_tile_xy(
        x=x, y=y, zoom_level=zl, in_crs="EPSG:4326", elev_type="dem10b"
    )
    assert isinstance(resps, TileData)
    assert isinstance(resps.tile_scope, TileScope)
    assert isinstance(resps.ary, np.ndarray)
    assert resps.ary.shape == (256, 256)
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    assert resps.tile_scope.x_min < m_x < resps.tile_scope.x_max
    assert resps.tile_scope.y_min < m_y < resps.tile_scope.y_max
    with pytest.raises(Exception):  # noqa: B017
        chiriin_drawer.fetch_elevation_tile_xy(
            x=x, y=y, zoom_level=zl + 1, in_crs="EPSG:4326", elev_type="dem10a"
        )


def test_fetch_elevation_tile_geometry_from_chiriin_drawer():
    """Test the get_elevation_tile_geometry function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_elevation_tile_geometry(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326", elev_type="dem10b"
    )
    assert isinstance(resps, list)
    x_idxs = []
    y_idxs = []
    for tile_data in resps:
        assert isinstance(tile_data, TileData)
        assert isinstance(tile_data.tile_scope, TileScope)
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256)
        assert tile_data.x_idx not in x_idxs
        assert tile_data.y_idx not in y_idxs
        x_idxs.append(tile_data.x_idx)
        y_idxs.append(tile_data.y_idx)

    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    x_distance = 10**100
    y_distance = 10**100
    for i in range(5, 15):
        resps = chiriin_drawer.fetch_elevation_tile_geometry(
            geometry=geom,
            zoom_level=i,
            in_crs="EPSG:4326",
            elev_type="dem10b",
        )
        assert isinstance(resps, list)
        tile_data = resps[0]
        _x_distance = tile_data.tile_scope.x_max - tile_data.tile_scope.x_min
        _y_distance = tile_data.tile_scope.y_max - tile_data.tile_scope.y_min
        assert _x_distance < x_distance
        assert _y_distance < y_distance
        x_distance = _x_distance
        y_distance = _y_distance
        x_min = min([tile_data.tile_scope.x_min for tile_data in resps])
        x_max = max([tile_data.tile_scope.x_max for tile_data in resps])
        y_min = min([tile_data.tile_scope.y_min for tile_data in resps])
        y_max = max([tile_data.tile_scope.y_max for tile_data in resps])
        assert x_min < m_x < x_max
        assert y_min < m_y < y_max


def test_fetch_elevation_tile_mesh_with_dem10b_from_chiriin_drawer():
    """Test the get_elevation_tile_mesh_with_dem10b function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_elevation_tile_mesh_with_dem10b(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    tile_data = resps[0]
    assert isinstance(tile_data, TileData)
    assert isinstance(tile_data.tile_scope, TileScope)
    assert isinstance(tile_data.ary, np.ndarray)
    assert tile_data.ary.shape == (256, 256)
    with pytest.raises(ValueError):
        chiriin_drawer.fetch_elevation_tile_mesh_with_dem10b(
            geometry=geom, zoom_level=15, in_crs="EPSG:4326"
        )


def test_fetch_elevation_tile_mesh_with_dem5a_from_chiriin_drawer():
    """Test the get_elevation_tile_mesh_with_dem5a function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 15
    resps = chiriin_drawer.fetch_elevation_tile_mesh_with_dem5a(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    tile_data = resps[0]
    assert isinstance(tile_data, TileData)
    assert isinstance(tile_data.tile_scope, TileScope)
    assert isinstance(tile_data.ary, np.ndarray)
    assert tile_data.ary.shape == (256, 256)
    with pytest.raises(ValueError):
        chiriin_drawer.fetch_elevation_tile_mesh_with_dem5a(
            geometry=geom, zoom_level=16, in_crs="EPSG:4326"
        )


def test_fetch_elevation_tile_mesh_with_dem5b_from_chiriin_drawer():
    """Test the get_elevation_tile_mesh_with_dem5b function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 15
    resps = chiriin_drawer.fetch_elevation_tile_mesh_with_dem5b(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    tile_data = resps[0]
    assert isinstance(tile_data, TileData)
    assert isinstance(tile_data.tile_scope, TileScope)
    if tile_data.ary is not None:
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256)
    with pytest.raises(ValueError):
        chiriin_drawer.fetch_elevation_tile_mesh_with_dem5b(
            geometry=geom, zoom_level=16, in_crs="EPSG:4326"
        )


def test_fetch_img_tile_xy_from_chiriin_drawer():
    """Test the get_img_tile_xy function."""
    x = 140.769399
    y = 39.769496
    zl = 14
    resps = chiriin_drawer.fetch_img_tile_xy(
        x=x, y=y, zoom_level=zl, in_crs="EPSG:4326", image_type="google_satellite"
    )
    assert isinstance(resps, TileData)
    assert isinstance(resps.tile_scope, TileScope)
    assert isinstance(resps.ary, np.ndarray)
    assert resps.ary.shape == (256, 256, 3)  # RGB image
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    assert resps.tile_scope.x_min < m_x < resps.tile_scope.x_max
    assert resps.tile_scope.y_min < m_y < resps.tile_scope.y_max
    with pytest.raises(Exception):  # noqa: B017
        chiriin_drawer.fetch_img_tile_xy(
            x=x, y=y, zoom_level=zl + 1, in_crs="EPSG:4326", image_type="osm"
        )


def test_fetch_img_tile_geometry_from_chiriin_drawer():
    """Test the get_img_tile_geometry function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_img_tile_geometry(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326", image_type="google_satellite"
    )
    assert isinstance(resps, list)
    for tile_data in resps:
        assert isinstance(tile_data, TileData)
        assert isinstance(tile_data.tile_scope, TileScope)
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256, 3)  # RGB image
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    x_min = min([tile_data.tile_scope.x_min for tile_data in resps])
    x_max = max([tile_data.tile_scope.x_max for tile_data in resps])
    y_min = min([tile_data.tile_scope.y_min for tile_data in resps])
    y_max = max([tile_data.tile_scope.y_max for tile_data in resps])
    assert x_min < m_x < x_max
    assert y_min < m_y < y_max


def test_fetch_img_tile_geometry_with_standard_map_from_chiriin_drawer():
    """Test the get_img_tile_geometry_with_standard_map function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_img_tile_geometry_with_standard_map(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    for tile_data in resps:
        assert isinstance(tile_data, TileData)
        assert isinstance(tile_data.tile_scope, TileScope)
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256, 3)  # RGB image
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    x_min = min([tile_data.tile_scope.x_min for tile_data in resps])
    x_max = max([tile_data.tile_scope.x_max for tile_data in resps])
    y_min = min([tile_data.tile_scope.y_min for tile_data in resps])
    y_max = max([tile_data.tile_scope.y_max for tile_data in resps])
    assert x_min < m_x < x_max
    assert y_min < m_y < y_max
    with pytest.raises(Exception):  # noqa: B017
        chiriin_drawer.fetch_img_tile_geometry_with_standard_map(
            geometry=geom, zoom_level=20, in_crs="EPSG:4326"
        )


def test_fetch_img_tile_geometry_with_photo_map_from_chiriin_drawer():
    """Test the get_img_tile_geometry_with_photo_map function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_img_tile_geometry_with_photo_map(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    for tile_data in resps:
        assert isinstance(tile_data, TileData)
        assert isinstance(tile_data.tile_scope, TileScope)
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256, 3)  # RGB image
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    x_min = min([tile_data.tile_scope.x_min for tile_data in resps])
    x_max = max([tile_data.tile_scope.x_max for tile_data in resps])
    y_min = min([tile_data.tile_scope.y_min for tile_data in resps])
    y_max = max([tile_data.tile_scope.y_max for tile_data in resps])
    assert x_min < m_x < x_max
    assert y_min < m_y < y_max
    with pytest.raises(Exception):  # noqa: B017
        chiriin_drawer.fetch_img_tile_geometry_with_photo_map(
            geometry=geom, zoom_level=20, in_crs="EPSG:4326"
        )


def test_fetch_img_tile_geometry_with_slope_map_from_chiriin_drawer():
    """Test the get_img_tile_geometry_with_slope_map function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_img_tile_geometry_with_slope_map(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    for tile_data in resps:
        assert isinstance(tile_data, TileData)
        assert isinstance(tile_data.tile_scope, TileScope)
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256, 3)  # RGB image
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    x_min = min([tile_data.tile_scope.x_min for tile_data in resps])
    x_max = max([tile_data.tile_scope.x_max for tile_data in resps])
    y_min = min([tile_data.tile_scope.y_min for tile_data in resps])
    y_max = max([tile_data.tile_scope.y_max for tile_data in resps])
    assert x_min < m_x < x_max
    assert y_min < m_y < y_max
    with pytest.raises(Exception):  # noqa: B017
        chiriin_drawer.fetch_img_tile_geometry_with_slope_map(
            geometry=geom, zoom_level=20, in_crs="EPSG:4326"
        )


def test_fetch_img_tile_geometry_with_google_satellite_from_chiriin_drawer():
    """Test the get_img_tile_geometry_with_google_satellite function."""
    x = 140.769399
    y = 39.769496
    geom = shapely.Point(x, y).buffer(0.003).envelope
    zl = 14
    resps = chiriin_drawer.fetch_img_tile_geometry_with_google_satellite(
        geometry=geom, zoom_level=zl, in_crs="EPSG:4326"
    )
    assert isinstance(resps, list)
    for tile_data in resps:
        assert isinstance(tile_data, TileData)
        assert isinstance(tile_data.tile_scope, TileScope)
        assert isinstance(tile_data.ary, np.ndarray)
        assert tile_data.ary.shape == (256, 256, 3)  # RGB image
    m_x, m_y = transform_xy(x=x, y=y, in_crs="EPSG:4326", out_crs="EPSG:3857")
    x_min = min([tile_data.tile_scope.x_min for tile_data in resps])
    x_max = max([tile_data.tile_scope.x_max for tile_data in resps])
    y_min = min([tile_data.tile_scope.y_min for tile_data in resps])
    y_max = max([tile_data.tile_scope.y_max for tile_data in resps])
    assert x_min < m_x < x_max
    assert y_min < m_y < y_max
    with pytest.raises(Exception):  # noqa: B017
        chiriin_drawer.fetch_img_tile_geometry_with_google_satellite(
            geometry=geom, zoom_level=20, in_crs="EPSG:4326"
        )


def test_fetch_geoid_height_from_chiriin_drawer():
    """Test the fetch_geoid_height function."""
    x = 140.769399
    y = 39.769496
    res = chiriin_drawer.fetch_geoid_height(x, y, in_crs="EPSG:4326")
    assert isinstance(res, float)
    assert 0 < res

    xs = [140.769399, 140.769500]
    ys = [39.769496, 39.769600]
    res = chiriin_drawer.fetch_geoid_height(xs, ys, in_crs="EPSG:4326")
    assert isinstance(res, list)
    assert all([isinstance(r, float) for r in res])
    assert all([r > 0 for r in res])
