import pyproj
import pytest
import shapely

from chiriin.paper import MapEditor

# 140.349215, 40.692805
# 140.348443, 40.693407

pnt1 = shapely.geometry.Point(140.349215, 40.692805)
pnt2 = shapely.geometry.Point(140.348443, 40.693407)
pnts = [pnt1, pnt2]
poly1 = pnt1.buffer(0.005).envelope
poly2 = pnt2.buffer(0.005).envelope
polys = [poly1, poly2]
m_poly = shapely.geometry.MultiPolygon([poly1, poly2])


@pytest.mark.parametrize(
    "geometry, in_crs, out_crs, paper_size",
    [
        (pnt1, "EPSG:4326", "EPSG:6678", "portrait_a4"),
        (pnt2, "EPSG:4326", None, "landscape_a4"),
        (pnts, "EPSG:4326", "EPSG:6691", "portrait_a3"),
        (poly1, "EPSG:4326", "EPSG:6678", "landscape_a3"),
        (poly2, "EPSG:4326", "EPSG:6678", "landscape_a4"),
        (polys, "EPSG:4326", "EPSG:6678", "landscape_a4"),
        (m_poly, "EPSG:4326", "EPSG:6678", "landscape_a4"),
    ],
)
def test_map_editor(geometry, in_crs, out_crs, paper_size):
    map_editor = MapEditor(
        geometry=geometry,
        in_crs=in_crs,
        out_crs=out_crs,
        paper_size=paper_size,
    )
    assert isinstance(map_editor.in_crs, pyproj.CRS)
    assert isinstance(map_editor.out_crs, pyproj.CRS)
    assert isinstance(map_editor.valid_scales, dict)
    geom_scope = map_editor.geom_scope
    min_scale = min(map_editor.valid_scales.keys())
    map_scope = map_editor.valid_scales[min_scale]
    assert map_scope.x_min <= geom_scope.x_min
    assert map_scope.y_min <= geom_scope.y_min
    assert geom_scope.x_max <= map_scope.x_max
    assert geom_scope.y_max <= map_scope.y_max
    with pytest.raises(Exception):
        map_editor = MapEditor(geometry, in_crs, paper_size="invalid size")
    with pytest.raises(Exception):
        map_editor = MapEditor(geometry, in_crs, "EPSG:6668", paper_size="portrait_a4")
    with pytest.raises(Exception):
        map_editor = MapEditor(
            [[geometry]], in_crs, "EPSG:6678", paper_size="portrait_a4"
        )


def test_x_from_map_editor():
    """Test the set_lims method of MapEditor."""
    map_editor = MapEditor(
        geometry=pnt1,
        in_crs="EPSG:4326",
        out_crs="EPSG:6678",
        paper_size="landscape_a4",
    )
    map_scope = map_editor.valid_scales.get(10_000)
    map_editor.set_lims(
        x_min=map_scope.x_min,
        y_min=map_scope.y_min,
        x_max=map_scope.x_max,
        y_max=map_scope.y_max,
    )
    # map_editor.ax.set_xlim(*map_scope, major_tick=500, major_grid=False, minor_grid=False)
    map_editor.remove_axis_grid()
    map_editor.add_icon_of_simple_compass(img_size=2.5)
    map_editor.add_icon_of_compass(img_size=2.5)
    map_editor.add_icon_of_true_north(img_size=2.5)
    map_editor.add_scale_txt(10_000)
    map_editor.add_basemap(*map_scope, map_name="slope", zl=15)
    map_editor.set_scope(*map_scope)
    with pytest.raises(Exception):
        map_editor.add_basemap(*map_scope, map_name="invalid_map", zl=15)
