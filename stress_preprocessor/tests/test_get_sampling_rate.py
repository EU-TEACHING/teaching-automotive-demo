import pandas as pd
from stress_preprocessor.utils.signal_processing_utils import get_sampling_rate


def test_get_sampling_rate():
    # Create test dataframes
    df1 = pd.DataFrame({"time": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    df2 = pd.DataFrame({"time": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    df3 = pd.DataFrame({"time": [0.0, 0.2, 0.4, 0.6]})

    # Calculate expected sampling rates
    expected_rates = [10.0, 100.0, 5.0]

    # Test the function with the test dataframes
    rates = get_sampling_rate([df1, df2, df3], "time")
    assert rates == expected_rates

    # Test that function raises ValueError for missing timestamp column
    try:
        get_sampling_rate([df1, df2, df3], "timestamp")
    except ValueError as e:
        assert str(e) == "Timestamp column 'timestamp' not found in dataframe."
