# よく使われるクラスと関数を直接インポート可能にする
from ._drawer import calculate_mean_slope_in_polygon, chiriin_drawer
from .config import MAG_DATA, XY, XYZ, ChiriinWebApi, Scope, TileData  # noqa: F401
from .formatter import (  # noqa: F401
    type_checker_crs,
    type_checker_float,
    type_checker_shapely,
)
from .geometries import degree_to_dms, dms_to_degree  # noqa: F401
from .mag import get_magnetic_declination  # noqa: F401
from .mesh import MeshCode  # noqa: F401
from .paper import MapEditor  # noqa: F401
from .semidynamic import SemiDynamic  # noqa: F401

map_editor = MapEditor
