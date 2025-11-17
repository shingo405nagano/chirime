import pytest
import shapely

from chiriin.config import TileInfo, TileScope, TileUrls
from chiriin.tile import (
    cut_off_points,
    download_tile_array,
    lonlat_to_tile_idx,
    search_tile_info_from_geometry,
    search_tile_info_from_xy,
)

tile_urls = TileUrls()


@pytest.mark.parametrize(
    "url, success",
    [(tile_urls.dem_10b.format(z=14, x=14624, y=6017), True)],
)
def test_download_tile_array(url, success):
    if success:
        ary = download_tile_array(url)
        assert ary.shape == (256, 256)
        assert ary.dtype == "float32"
    else:
        with pytest.raises(Exception):  # noqa: B017
            download_tile_array(url)


@pytest.mark.parametrize(
    "lon, lat, zoom_level, in_crs, x_idx, y_idx",
    [
        (140.087099, 40.504665, 14, "EPSG:4326", 14567, 6172),
        (140.087099, 40.504665, 15, "EPSG:4326", 29135, 12345),
        (140.087099, 40.504665, 16, "EPSG:4326", 58270, 24690),
        (139.087099, 35.504665, 12, "EPSG:4326", 3630, 1615),
        (139.087099, 35.504665, 15, "EPSG:4326", 29044, 12923),
    ],
)
def test_lonlat_to_tile_idx(lon, lat, zoom_level, in_crs, x_idx, y_idx):
    """Test the lonlat_to_tile_idx function."""
    x, y = lonlat_to_tile_idx(lon, lat, zoom_level, in_crs=in_crs)
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert x == x_idx
    assert y == y_idx


def test_cut_off_points():
    """Test the cut_off_points function."""
    preview = 0
    preview_x = 0
    preview_y = 0
    for zl in range(0, 20):
        points = cut_off_points(zl)
        assert isinstance(points, dict)
        assert "Y" in points
        assert "X" in points
        assert preview_x <= len(points["X"])
        assert preview_y <= len(points["Y"])
        assert preview < len(points["X"]) * len(points["Y"])
        preview = len(points["X"]) * len(points["Y"])
        preview_x = len(points["X"])
        preview_y = len(points["Y"])

    with pytest.raises(Exception):  # noqa: B017
        cut_off_points(-1)

    with pytest.raises(Exception):  # noqa: B017
        cut_off_points("invalid")


def test_search_tile_info_from_xy():
    """Test the search_tile_info_from_xy function."""
    lon = 140.3158733
    lat = 38.3105495
    crs = "EPSG:4326"
    preview_x_resol = 0
    preview_y_resol = 0
    for zl in sorted(list(range(0, 20)), reverse=True):
        tile_info = search_tile_info_from_xy(lon, lat, zl, in_crs=crs)
        assert isinstance(tile_info, TileInfo)
        assert preview_x_resol < tile_info.x_resolution
        assert preview_y_resol < tile_info.y_resolution
        preview_x_resol = tile_info.x_resolution
        preview_y_resol = tile_info.y_resolution


def test_search_tile_info_from_geometry():
    """Test the search_tile_info_from_geometry function."""
    geom = shapely.Point(140.3158733, 38.3105495).buffer(0.1).envelope
    crs = "EPSG:4326"
    tile_geoms = []
    for zl in sorted(list(range(0, 18)), reverse=True):
        tile_info_list = search_tile_info_from_geometry(geom, zl, in_crs=crs)
        assert isinstance(tile_info_list, list)
        assert all(isinstance(ti, TileInfo) for ti in tile_info_list)
        for tl in tile_info_list:
            scope = shapely.box(*tl.tile_scope)
            tile_geoms.append(scope)

    result_tile_scope = TileScope(*shapely.union_all(tile_geoms).bounds)
    geom_scope = TileScope(*geom.bounds)
    assert result_tile_scope.x_min <= geom_scope.x_min
    assert result_tile_scope.y_min <= geom_scope.y_min
    assert geom_scope.x_max <= result_tile_scope.x_max
    assert geom_scope.y_max <= result_tile_scope.y_max
