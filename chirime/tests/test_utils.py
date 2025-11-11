import numpy as np
import pytest

from chiriin.utils import dimensional_count


@pytest.mark.parametrize(
    "iterable, expected",
    [
        ("100", 0),
        (100, 0),
        (100.0, 0),
        ((100, 200), 1),
        ([100, 200], 1),
        ([[100, 200], [300, 400]], 2),
        ([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], 3),
        (np.array([1, 2, 3]), 1),
        (np.array([[1, 2], [3, 4]]), 2),
    ],
)
def test_dimensional_count(iterable, expected):
    """Test dimensional_count function."""
    result = dimensional_count(iterable)
    assert result == expected
