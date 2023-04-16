import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from stress_preprocessor.utils.preprocessing_utils import validate_timestamps


@pytest.fixture
def sample_data():
    df1 = pd.DataFrame({
        "time": [0.0, 0.1, 0.2, 0.4, 0.5],
        "value": [1, 2, 3, 4, 5]
    })
    df2 = pd.DataFrame({
        "time": [0.0, 0.1, 0.3, 0.4, 0.5],
        "value": [6, 7, 8, 9, 10]
    })

    return [df1, df2]


def test_validate_timestamps(sample_data):
    validated_data = validate_timestamps(sample_data, "time", 10)
    expected_df1 = pd.DataFrame({
        "time": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "value": [1, 2, 3, np.nan, 4, 5]
    }).astype({"value": float})

    expected_df2 = pd.DataFrame({
        "time": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "value": [6, 7, np.nan, 8, 9, 10]
    }).astype({"value": float})

    assert_frame_equal(validated_data[0], expected_df1)
    assert_frame_equal(validated_data[1], expected_df2)
