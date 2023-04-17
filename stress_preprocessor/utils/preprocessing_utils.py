from decimal import Decimal
import logging
from typing import List

import numpy as np
import pandas as pd


# Define a function to round a value to 2 decimal points
def round_decimal(val):
    return Decimal(val).quantize(Decimal('0.01'))


def clean_duplicates(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Removes duplicate rows from a list of pandas DataFrames.

    Args:
        dfs (List[pandas.DataFrame]): A list of DataFrames to clean.

    Returns:
        List[pandas.DataFrame]: A list of cleaned DataFrames.
    """
    cleaned_dfs = []

    for df in dfs:
        # Remove duplicates
        cleaned_df = df.drop_duplicates()
        cleaned_dfs.append(cleaned_df)

    return cleaned_dfs


def validate_timestamps(dfs: List[pd.DataFrame], time_col: str, sampling_rate: float) -> List[pd.DataFrame]:
    """
    Validates the timestamps in a list of DataFrames to ensure that they are monotonically increasing at the specified
    sampling rate. If some timestamps are missing, they will be added with missing values for the rest of the columns.

    Args:
        dfs: A list of pandas DataFrames.
        time_col: The name of the column containing the timestamps.
        sampling_rate: The expected sampling rate (in Hz).

    Returns:
        A list of pandas DataFrames with any missing rows added and NaN values filled for the other columns.
    """
    validated_dfs = []
    for df in dfs:
        # Convert to Decimal to avoid errors in merging
        df[time_col] = df[time_col].apply(round_decimal)
        df = df.sort_values(time_col)
        # Check for missing timestamps
        time_diff = df[time_col].diff().fillna(0)
        expected_time_diff = 1 / sampling_rate
        if (time_diff.abs() > expected_time_diff).any():
            # Missing timestamps found; fill them in with NaN values

            # Create the complete timestamps as a range (start, stop, step)
            complete_time = pd.Series(np.arange(float(df[time_col].min()), float(df[time_col].max()) + expected_time_diff, expected_time_diff))
            complete_df = pd.DataFrame({time_col: complete_time})
            # Use Decimal to avoid rounding errors in the merging key column
            complete_df[time_col] = complete_df[time_col].apply(round_decimal)

            # Merging will add NULL for any extra timestamp added
            df = pd.merge(complete_df, df, on=time_col, how='outer')

            # Convert the timestamp column back to numeric
            df[time_col] = df[time_col].apply(lambda x: float(x))

        validated_dfs.append(df)

    return validated_dfs


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Prefixes every column name in a DataFrame with a given string, unless the column name is already prefixed with it.

    Args:
        df (pd.DataFrame): Input DataFrame to modify
        prefix (str): String to use as a prefix for the column names

    Returns:
        pd.DataFrame: DataFrame with modified column names
    """
    new_cols = []
    for col in df.columns:
        if not col.startswith(prefix):
            new_cols.append(f"{prefix}{col}")
        else:
            new_cols.append(col)
    return df.rename(columns=dict(zip(df.columns, new_cols)))


def compute_diff(dataframes: List[pd.DataFrame], columns: List[str]) -> List[pd.DataFrame]:
    """
    Computes the first order differences for the specified columns in a list of dataframes and
    adds them as new columns in the dataframes.

    Args:
        dataframes: A list of pandas dataframes to compute the differences for.
        columns: A list of column names to compute the differences for.

    Returns:
        A list of pandas dataframes, each with the new columns representing the first order differences.
    """

    diff_dfs = []

    for df in dataframes:
        for col in columns:
            col_diff = col + '_diff'
            df[col_diff] = df[col].diff()
            df.at[0, col_diff] = 0  # Replace NaN with 0 for the first element
        diff_dfs.append(df)

    return diff_dfs


def resample_dataframe(df: pd.DataFrame, timestamp_col: str, value_col: str, group_col: str, participant_col: str,
                       target_freq: str) -> pd.DataFrame:
    """Resample a time series DataFrame to a specified frequency.

    Args:
        df (pandas.DataFrame): The DataFrame to resample.
        timestamp_col (str): The name of the column containing the timestamp data.
        value_col (str): The name of the column containing the data to be resampled.
        target_freq (str): The target frequency to which the DataFrame should be resampled.
            Must be a valid pandas frequency string (e.g., '1H', '1D', '1W').

    Returns:
        pandas.DataFrame: The resampled DataFrame with a DateTimeIndex and missing values filled forward.

    Raises:
        ValueError: If the specified timestamp column is not a valid datetime column.

    """
    # Set timestamp column as DataFrame index
    df.set_index(timestamp_col, inplace=True)

    # Resample DataFrame to target frequency
    resampled_df = df.resample(target_freq).agg({participant_col: 'first',
                                                 group_col: 'first',
                                                 value_col: 'mean'})

    # Reset the index to timestamp_col column name
    resampled_df.reset_index(inplace=True)
    resampled_df.rename(columns={'index': 'timestamp_col'}, inplace=True)

    return resampled_df


def check_valid_timestamps(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Checks if all timestamps in a pandas DataFrame are valid.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        timestamp_col (str): The name of the column containing timestamps.

    Returns:
        pandas.DataFrame: The DataFrame with invalid timestamps converted to NaT.
    """

    # Check if all values in the timestamp column can be parsed as pandas timestamps
    try:
        pd.to_datetime(df[timestamp_col], errors='raise')
        return df
    except ValueError as e:
        invalid_rows = df[pd.to_datetime(df[timestamp_col], errors='coerce').isna()].index
        logging.info(f"Invalid timestamps found in rows: {list(invalid_rows)}")
        df.loc[invalid_rows, timestamp_col] = pd.NaT
        return df


def reset_timer(df: pd.DataFrame, group_col: str, timestamp_col: str) -> pd.DataFrame:
    """Reset the timer to 0 every time a new scenario begins.

    Args:
        df (pd.DataFrame): Input DataFrame.
        group_col (str): Name of the column that contains the groups.
        timestamp_col (str): Name of the column that contains the time values.

    Returns:
        pd.DataFrame: The input DataFrame with the timer reset to 0 at the beginning of each group.

    """
    # Sort the DataFrame by ts_corr
    df = df.sort_values(timestamp_col)

    # Create a new column 'ts_corr' that tracks the time elapsed since
    # the beginning of the scenario
    df[timestamp_col] = df.groupby(group_col)[timestamp_col].diff().fillna(pd.Timedelta(seconds=0))

    # For each scenario, reset the timer to 0 at the beginning
    df[timestamp_col] = df.groupby(group_col, group_keys=False)[timestamp_col].apply(lambda x: x.cumsum())

    return df


def adjust_scenario_length(df: pd.DataFrame, group_col: str, timestamp_col: str, value_col: str,
                           policy: str) -> pd.DataFrame:
    """
    Transforms the input DataFrame so that all scenario groups are of the same length.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing scenario data.
        group_col (str): The name of the column containing the scenario group identifiers.
        timestamp_col (str): The name of the column containing the timestamps for the scenario data.
        value_col (str): The name of the column containing scenario data to transform.
        policy (str): The policy to use for scenario transformation. Must be 'min' or 'max'.

    Returns:
        pd.DataFrame: A pandas DataFrame with transformed scenario data.
    """
    if policy not in ['min', 'max']:
        raise ValueError(f"Invalid policy '{policy}'. Policy must be 'min' or 'max'.")

    # df = reset_timer(df, group_col, timestamp_col)

    if policy == 'min':
        min_duration = df.groupby(group_col).size().min()
        df = df.groupby(group_col).head(min_duration)
        df = df.loc[:, [group_col, timestamp_col, value_col]]
        df = df.reset_index(drop=True)

    elif policy == 'max':
        max_duration = df.groupby(group_col).size().max()
        pad_rows = []
        for group, data in df.groupby(group_col):
            num_rows = len(data)
            if num_rows < max_duration:
                pad_rows.append(pd.DataFrame({group_col: [group] * (max_duration - num_rows),
                                              timestamp_col: [data[timestamp_col].iloc[-1]] * (max_duration - num_rows),
                                              value_col: [0] * (max_duration - num_rows)}))
        df = pd.concat([df, *pad_rows], ignore_index=True)

        df = df.groupby([group_col, timestamp_col])[value_col].first().unstack()

        # Convert wide format to long format
        df = df.stack().reset_index()
        df.columns = [group_col, timestamp_col, value_col]

    return df


def pad_strings(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Check whether a column in a Pandas DataFrame contains strings less than 4 characters,
    and pads each string with zeros at the beginning until it's at least 4 characters long.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to check and pad.

    Returns:
        pd.DataFrame: The updated DataFrame.

    Raises:
        ValueError: If the column does not contain strings.
    """
    # Check if the column contains strings
    if df[col_name].dtype != 'object':
        raise ValueError('Column must contain strings')

    # Check if any strings are less than 4 characters
    if (df[col_name].str.len() < 4).any():
        # Pad each string with zeros until it's at least 4 characters long
        df[col_name] = df[col_name].apply(lambda x: x.zfill(4))

    return df


def create_duration_column(scenario_df) -> pd.DataFrame:
    """Calculates the duration of each scenario in a pandas dataframe using the 'start_time' and 'end_time' columns.

    Args:
        scenario_df (pandas.DataFrame): A pandas dataframe containing the 'start_time' and 'end_time' columns.

    Returns:
        pandas.DataFrame: A pandas dataframe with the 'duration' fixed to the time difference between 'start_time' and
        'end_time' in timedelta format.

    """
    # Drop the initial duration column
    scenario_df.drop(columns=['duration'], inplace=True)

    scenario_df['duration'] = scenario_df['end_time'] - scenario_df['start_time']
    return scenario_df


def merge_scenario_and_signal_dataframes(scenario_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two dataframes based on a logic that involves matching rows from both dataframes based on their 'prob_id',
    'start_time', and 'end_time' columns. Rows in the resulting dataframe are filtered to exclude rows where the
    timestamp 'ts_corr' is not within the range defined by the 'start_time' and 'end_time' columns. Rows with a
    timestamp earlier than the earliest 'start_time' are assigned a 'scenario_id' of 0.
    The resulting dataframe is sorted by 'prob_id', 'ts_corr', and 'scenario_id'.

    Args:
        scenario_df (pd.DataFrame): The scenario dataframe to be merged.
        signal_df (pd.DataFrame): The signal dataframe to be merged.

    Returns:
        pd.DataFrame: The merged dataframe.

    Raises:
        ValueError: If 'prob_id' column in both dataframes does not have the same data type.

    """
    # Sort the dataframes by the relevant time columns
    sorted_scenario_df = scenario_df.sort_values(by=['start_time'])
    sorted_signal_df = signal_df.sort_values(by=['ts_corr'])

    # Create a new dataframe to store the merged data
    merged_df = pd.DataFrame(columns=['prob_id', 'ts_corr', 'scenario_id'])

    # Loop over each scenario in the scenario dataframe
    for idx, row in sorted_scenario_df.iterrows():
        # Select the relevant rows from the signal dataframe based on the scenario start and end times
        mask = (sorted_signal_df['ts_corr'] >= row['start_time']) & (sorted_signal_df['ts_corr'] <= row['end_time'])
        selected_signal_df = sorted_signal_df.loc[mask].copy()

        # Set the scenario ID for the selected rows
        selected_signal_df['scenario_id'] = row['scenario_id']

        # If this is the first scenario, set the scenario ID for any earlier rows to 0
        if idx == 0:
            mask = sorted_signal_df['ts_corr'] < row['start_time']
            earlier_signal_df = sorted_signal_df.loc[mask].copy()
            earlier_signal_df['scenario_id'] = 0
            selected_signal_df = pd.concat([earlier_signal_df, selected_signal_df], axis=0)

        # Add the selected rows to the merged dataframe
        merged_df = pd.concat([merged_df, selected_signal_df], axis=0)

    # Check if 'prob_id' column in both dataframes has the same data type
    if scenario_df['prob_id'].dtype != signal_df['prob_id'].dtype:
        raise ValueError("The 'prob_id' column in both dataframes must have the same data type.")

    merged_df['scenario_id'] = merged_df['scenario_id'].astype(str)

    return merged_df


def resample_dataframe(df: pd.DataFrame, timestamp_col: str, value_col: str, group_col: str, participant_col: str,
                       target_freq: str) -> pd.DataFrame:
    """Resample a time series DataFrame to a specified frequency.

    Args:
        df (pandas.DataFrame): The DataFrame to resample.
        timestamp_col (str): The name of the column containing the timestamp data.
        value_col (str): The name of the column containing the data to be resampled.
        target_freq (str): The target frequency to which the DataFrame should be resampled.
            Must be a valid pandas frequency string (e.g., '1H', '1D', '1W').

    Returns:
        pandas.DataFrame: The resampled DataFrame with a DateTimeIndex and missing values filled forward.

    Raises:
        ValueError: If the specified timestamp column is not a valid datetime column.

    """
    # Set timestamp column as DataFrame index
    df.set_index(timestamp_col, inplace=True)

    # Resample DataFrame to target frequency
    resampled_df = df.resample(target_freq).agg({participant_col: 'first',
                                                 group_col: 'first',
                                                 value_col: 'mean'})

    # Reset the index to timestamp_col column name
    resampled_df.reset_index(inplace=True)
    resampled_df.rename(columns={'index': 'timestamp_col'}, inplace=True)

    return resampled_df
