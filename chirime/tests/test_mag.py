import pytest

from chiriin.geometries import degree_to_dms
from chiriin.mag import get_magnetic_declination
from chiriin.tests.data.data import PREF_DF


@pytest.mark.parametrize(
    "lon, lat, is_dms, idx",
    [
        (
            row["longitude"] if i < 40 else degree_to_dms(row["longitude"]),
            row["latitude"] if i < 40 else degree_to_dms(row["latitude"]),
            False if i < 40 else True,
            i,
        )
        for i, row in PREF_DF.iterrows()
    ],
)
def test_get_magnetic_declination(lon, lat, is_dms, idx):
    """Test the get_magnetic_declination function with various coordinates."""
    result = get_magnetic_declination(lon, lat, is_dms)
    assert isinstance(result, float), "Result should be a float"

    if idx == 0:
        # Waringsのテストは、最初の行でのみ行う
        get_magnetic_declination(146.00, 40.00, is_dms=False)
