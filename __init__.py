"""
Chiriin - 国土地理院のAPIやパラメーターファイルを利用するモジュール
"""

from .chirime import (
    MAG_DATA,
    XY,
    XYZ,
    ChiriinWebApi,
    MapEditor,
    MeshCode,
    Scope,
    SemiDynamic,
    TileData,
    calculate_mean_slope_in_polygon,
    chiriin_drawer,
    degree_to_dms,
    dms_to_degree,
    get_magnetic_declination,
    map_editor,
    type_checker_crs,
    type_checker_float,
    type_checker_shapely,
)
from .geomesh import *

__version__ = "1.0.0"
__all__ = [
    "geomesh"
    # Drawer
    "chiriin_drawer",
    "map_editor",
    # Config
    "MAG_DATA",
    "XY",
    "XYZ",
    "Scope",
    "TileData",
    "ChiriinWebApi",
    # Features
    "calculate_mean_slope_in_polygon",
    # Formatter
    "type_checker_float",
    "type_checker_crs",
    "type_checker_shapely",
    # Geometries
    "dms_to_degree",
    "degree_to_dms",
    # Mesh
    "MeshCode",
    # Mag
    "get_magnetic_declination",
    # Semidynamic
    "SemiDynamic",
    # Paper
    "MapEditor",
]
