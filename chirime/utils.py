from typing import Union

import numpy as np
import pandas as pd

UniqueIterable = Union[tuple, list, np.ndarray, pd.Series]


def dimensional_count(value: UniqueIterable) -> int:
    """
    ## Description:
        オブジェクトがどの程度の次元を持つリストであるかを測定する関数。
    Args:
        value (tuple | list | np.ndarray | pd.Series):
            測定したい値。リスト、タプル、NumPy配列、またはPandas Seriesを受け入れます。
            それ以外の型（str, int, floatなど）は0次元と見なされます。
    Returns:
        int:
            測定された次元の数を返します。
            - 0: 値がリストではない場合（str, int, floatなど）。
            - 1: 1次元の場合。
            - 2: 2次元の場合。
            - 3: 3次元の場合。
            ...
    Examples:
        >>> dimensional_measurement(1)
        0
        >>> dimensional_measurement('a')
        0
        >>> dimensional_measurement([1, 2, 3])
        1
        >>> dimensional_measurement([[1, 2, 3], [4, 5, 6]])
        2
        >>> dimensional_measurement([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        3
    """
    if isinstance(value, UniqueIterable):
        try:
            # Convert numpy arrays to lists
            value = value.tolist()  # type: ignore
        except:  # noqa: E722
            try:
                # Convert pandas Series to lists
                value = value.to_list()  # type: ignore
            except:  # noqa: E722
                pass
        return 1 + max(dimensional_count(item) for item in value) if value else 1  # type: ignore
    else:
        return 0
