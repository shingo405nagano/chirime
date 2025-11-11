import math
import warnings

from .config import MAG_DATA
from .formatter import type_checker_float
from .geometries import dms_to_degree
from .mesh import MeshCode


@type_checker_float(arg_index=0, kward="lon")
@type_checker_float(arg_index=1, kward="lat")
def get_magnetic_declination(lon: float, lat: float, is_dms: bool = False) -> float:
    """
    ## Description:
        磁気偏角を取得する関数
    ## Args:
        lon (float):
            経度（10進法経緯度 | 経緯度（度分秒））
        lat (float):
            緯度（10進法経緯度 | 経緯度（度分秒））
        is_dms (bool):
            度分秒経緯度であるかどうか
    ## Returns:
        float:
            磁気偏角
    """
    if is_dms:
        lon = dms_to_degree(lon, decimal_obj=False)  # type: ignore
        lat = dms_to_degree(lat, decimal_obj=False)  # type: ignore
    mesh_code = MeshCode(lon, lat)
    mag = MAG_DATA.get(mesh_code.secandary_mesh_code, None)
    if mag is None:
        h = "-" * 100
        warnings.warn(
            "\nMagnetic declination data not found for mesh code.\nPlease check if "
            "the coordinates are within Japan or if the mesh code is correct.\n"
            f"Longitude: {lon}, Latitude: {lat}, "
            f"Mesh Code: {mesh_code.secandary_mesh_code}\n" + h,
            stacklevel=2,
        )
        return math.nan
    return mag
