import pytest

from chiriin.mesh import MeshCode


@pytest.mark.parametrize(
    "lon, lat, is_dms, mc1d, mc2d, mc3d, mc4d",
    [
        # 北海道札幌市
        (141.354368, 43.062072, False, "6441", "644142", "64414278", "644142781"),
        # 青森青森市
        (140.7469, 40.8227, False, "6140", "614015", "61401589", "614015894"),
        (1404448.84, 404921.71999, True, "6140", "614015", "61401589", "614015894"),
        # 秋田秋田市
        (140.1035, 39.72, False, "5940", "594040", "59404068", "594040681"),
        # 新潟県新潟市
        (139.036725, 37.916094, False, "5639", "563960", "56396092", "563960924"),
        # 東京東京都千代田区
        (139.753561, 35.693857, False, "5339", "533946", "53394630", "533946301"),
        # 京都府京都市
        (135.767884, 35.011607, False, "5235", "523546", "52354611", "523546111"),
    ],
)
def test_mesh_code(lon, lat, is_dms, mc1d, mc2d, mc3d, mc4d):
    mesh_code = MeshCode(lon=lon, lat=lat, is_dms=is_dms)
    assert mesh_code.first_mesh_code == mc1d
    assert mesh_code.secandary_mesh_code == mc2d
    assert mesh_code.standard_mesh_code == mc3d
    assert mesh_code.half_mesh_code == mc4d
    print(mesh_code)  # This is not checked, but can be used for debugging
