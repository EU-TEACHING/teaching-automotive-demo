import numpy as np
import pandas as pd


class SlidingWindow:
    def __init__(self, baseline_data):
        """
        Initialize the SlidingWindow object.

        Args:
            baseline_data (array): An array with the baseline data.
        """
        self.array = baseline_data

    def __call__(self, new_data):
        """
        Read dataframe from local directory, convert it to array, apply sliding window,
        and update with new arrays from streaming data.

        Args:
            new_data (numpy.array): New data from streaming.

        Returns:
            numpy.array: Array of sliding windows.
        """

        # Remove oldest entry from array
        self.array = np.delete(self.array, len(new_data), axis=0)
        # Append new data
        self.array = np.vstack((self.array, new_data))

        return self.array
