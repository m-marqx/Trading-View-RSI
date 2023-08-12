from typing import Literal

import numpy as np
import pandas as pd

def _rma_pandas(
    source: pd.Series,
    length: int,
    **kwargs
) -> pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the pandas EWM (Exponential
        Weighted Moving Average) function.

    Returns:
    --------
    pandas.Series
        The calculated RMA time series data.

    Note:
    -----
    The first values are different from the TradingView RMA.
    """
    sma = source.rolling(window=length, min_periods=length).mean()[:length]
    rest = source[length:]
    return (
        pd.concat([sma, rest])
        .ewm(alpha=1 / length, **kwargs)
        .mean()
    ).rename("RMA")

