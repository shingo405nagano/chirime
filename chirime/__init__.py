from ._drawer import calculate_mean_slope_in_polygon, chirime
from .config import MAG_DATA, XY, XYZ, ChiriinWebApi, Scope, TileData
from .formatter import (
    type_checker_crs,
    type_checker_float,
    type_checker_shapely,
)
from .geometries import degree_to_dms, dms_to_degree
from .mag import get_magnetic_declination
from .paper import MapEditor
from .semidynamic import SemiDynamic

map_editor = MapEditor

__version__ = "0.1.1"

__all__ = [
    "calculate_mean_slope_in_polygon",
    "chirime",
    "MAG_DATA",
    "XY",
    "XYZ",
    "ChiriinWebApi",
    "Scope",
    "TileData",
    "type_checker_crs",
    "type_checker_float",
    "type_checker_shapely",
    "degree_to_dms",
    "dms_to_degree",
    "get_magnetic_declination",
    "MapEditor",
    "SemiDynamic",
]
