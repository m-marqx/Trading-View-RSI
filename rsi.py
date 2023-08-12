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

def _rma_python(
    source: pd.Series,
    length: int
) -> pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data using pure python.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.

    Returns:
    --------
    pd.Series
        The calculated RMA time series data.

    Note:
    -----
    The pure python version is the only one with precision in the
    initial RMA values. However, with the simple RMA version,
    both pandas and python versions will yield the same precision
    in initial values.
    """
    alpha = 1 / length
    source_pd = _rma_pandas(source, length)[:length]
    source_values = source[length:].to_numpy().tolist()

    rma_value = float(source_pd.dropna().iloc[0])

    rma_list = [rma_value] + [
        rma_value := alpha * source_value + ((1 - alpha) * rma_value)
        for source_value in source_values
    ]

    rma_series = pd.Series(
        rma_list,
        name="RMA",
        index=source[length - 1:].index
    )

    return rma_series

def rma(
    source: pd.Series,
    length: int,
    method: Literal["numpy", "pandas"] = "numpy"
) -> np.ndarray | pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.
    method : {"numpy", "pandas"}, optional
        The method to use for calculating the RMA, by default "numpy".

    Returns:
    --------
    np.ndarray or pandas.Series
        The calculated RMA time series data.
    """
    match method:
        case "numpy":
            return _rma_python(source, length)
        case "pandas":
            return _rma_pandas(source, length)
        case _:
            raise TypeError("method must be 'numpy' or 'pandas'")

