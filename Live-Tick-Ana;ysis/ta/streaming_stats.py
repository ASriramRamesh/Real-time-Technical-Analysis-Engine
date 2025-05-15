import numpy as np
from collections import deque
import logging


class StreamingStats:
    def __init__(self, window_size=5, only_independent=False):
        """
        Initialize the streaming statistics calculator.

        Args:
            window_size (int): Number of data points to maintain in rolling window
        """
        self.window_size = window_size
        self.values_y = deque(maxlen=window_size)
        self.values_x = deque(maxlen=window_size)
        # For Nifty we only need independent values as it cannot be compared to another sector.
        self.only_independent = only_independent
        self.stats = {
            "Independent": {
                "slope": None,
                "intercept": None,
                "pearson": None,
                "r_squared": None,
            },
            "Dependent": {
                "slope": None,
                "intercept": None,
                "pearson": None,
                "r_squared": None,
            },
        }

    def update(self, new_value_y, new_value_x=None):
        """
        Update with new data point and calculate all statistics.

        Args:
            new_value (float): New data point

        Returns:
            dict: Dictionary containing all calculated statistics
        """
        # Add new value to the window
        self.values_y.append(new_value_y)
        if not self.only_independent:
            self.values_x.append(new_value_x)

        # Only calculate if we have at least 5 points
        if len(self.values_y) < 5:
            return

        # Convert deque to numpy array
        y = np.array(self.values_y)
        x = np.arange(len(y))

        # Calculate linear regression using numpy's polyfit
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate Pearson's correlation coefficient
        pearson = np.corrcoef(x, y)[0, 1]

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        self.stats["Independent"]["slope"] = slope
        self.stats["Independent"]["intercept"] = intercept
        self.stats["Independent"]["pearson"] = pearson
        self.stats["Independent"]["r_squared"] = r_squared

        # If we want the sector or Nifty independent values.
        if not self.only_independent:
            x = np.array(self.values_x)

            # Calculate correlation and regression
            slope, intercept = np.polyfit(x, y, 1)
            pearson = np.corrcoef(x, y)[0, 1]

            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)

            r_squared = 1 - (ss_res / ss_tot)

            self.stats["Dependent"]["slope"] = slope
            self.stats["Dependent"]["intercept"] = intercept
            self.stats["Dependent"]["pearson"] = pearson
            self.stats["Dependent"]["r_squared"] = r_squared

    def get_current_values(self):
        """Return current values in the window"""
        return list(self.values_y)


def demonstrate_streaming():
    # Initialize calculators for both indices
    calculator_banknifty = StreamingStats(window_size=5)
    calculator_nifty = StreamingStats(window_size=5)

    logging.basicConfig(filename="output.log", level=logging.INFO)

    nifty = [
        4.02,
        1.89,
        1.59,
        8.08,
        -0.54,
    ]

    # Sample data
    bank_nifty = [
        37.32,
        53.93,
        40.94,
        8.26,
        -21.76,
    ]

    logging.info("Analyzing Bank Nifty vs Nifty relationship...")

    def calculate_cross_metrics(x_values, y_values):
        """Calculate metrics between two series"""
        if len(x_values) < 2:
            return None

        x = np.array(x_values)
        y = np.array(y_values)

        # Calculate correlation and regression
        slope, intercept = np.polyfit(x, y, 1)
        pearson = np.corrcoef(x, y)[0, 1]

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            "slope": slope,
            "intercept": intercept,
            "pearson": pearson,
            "r_squared": r_squared,
        }

    for i, (bn_value, n_value) in enumerate(zip(bank_nifty, nifty), 1):
        # Update both calculators
        stats_bn = calculator_banknifty.update(bn_value)
        stats_nifty = calculator_nifty.update(n_value)

        # Get current windows
        bn_window = calculator_banknifty.get_current_values()
        nifty_window = calculator_nifty.get_current_values()

        # Only proceed if we have enough data points
        if len(bn_window) >= 5:
            # Calculate cross-metrics using Nifty as x and Bank Nifty as y
            cross_metrics = calculate_cross_metrics(nifty_window, bn_window)

            logging.info(f"\nAfter {i} points:")
            logging.info(f"Window Bank Nifty (y): {bn_window}")
            logging.info(f"Window Nifty (x): {nifty_window}")
            logging.info("\nRelationship Metrics (Nifty as X, Bank Nifty as Y):")
            logging.info(f"Slope: {cross_metrics['slope']:.4f}")
            logging.info(f"Y-Intercept: {cross_metrics['intercept']:.4f}")
            logging.info(f"Pearson's Coefficient: {cross_metrics['pearson']:.4f}")
            logging.info(f"R-squared: {cross_metrics['r_squared']:.4f}")

            # Individual series metrics
            logging.info("\nIndividual Series Metrics:")
            logging.info("Bank Nifty Trend:")
            logging.info(f"Slope: {stats_bn['slope']:.4f}")
            logging.info(f"R-squared: {stats_bn['r_squared']:.4f}")
            logging.info("\nNifty Trend:")
            logging.info(f"Slope: {stats_nifty['slope']:.4f}")
            logging.info(f"R-squared: {stats_nifty['r_squared']:.4f}")
            logging.info("-" * 50)


if __name__ == "__main__":
    demonstrate_streaming()
