# Warning

This repo has been archived because it will now be maintained by the [TradingView Indicators](https://github.com/m-marqx/TradingView-Indicators) library.

## Objective
The purpose of this repository is to serve as a model for RSI calculation. When using the RMA calculation with the `.ewm` method in pandas, the initial results differ from those in TradingView. To address this, the solution involves manual calculation based on the Pinescript formula:

## RSI Construction

```python
pine_rsi(x, y) => 
    u = math.max(x - x[1], 0) // upward ta.change
    d = math.max(x[1] - x, 0) // downward ta.change
    rs = ta.rma(u, y) / ta.rma(d, y)
    res = 100 - 100 / (1 + rs)
    res
```

For consistency, numpy and pandas were utilized to vectorize values in line with the RSI calculation. The approach employed is as follows:

```python
def py_rsi(x, y):
    u = pd.Series(np.maximum(x - x.shift(1), 0.0)).dropna()
    d = pd.Series(np.maximum(x.shift(1) - x, 0.0)).dropna()
    rs = rma(u, y) / rma(d, y)
    res = 100 - (100 / (1 + rs))
    return res.rename("RSI")
```

For clarity, variable and parameter names were made more descriptive, I followed the naming convention used in TradingView:

```python
def rsi(source: pd.Series, length: int) -> pd.Series:
    upward_diff = pd.Series(
        np.maximum(source - source.shift(1), 0.0)
    ).dropna()

    downward_diff = pd.Series(
        np.maximum(source.shift(1) - source, 0.0)
    ).dropna()

    relative_strength = rma(upward_diff, length) / rma(downward_diff, length)

    rsi_series = 100 - (100 / (1 + relative_strength))
    return rsi_series.rename("RSI")
```

## RMA Construction


In the RMA tradingview documentation, the alpha is "1/length" as seen below:

```python
pine_rma(src, length) =>
    alpha = 1/length
    sum = 0.0
    sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])
plot(pine_rma(close, 15))
``` 

To calculate the RMA, my initial approach was to utilize `.ewm(alpha=1/length)` with the mentioned alpha value. 

```python
def _rma_pandas(source: pd.Series, length: int, **kwargs) -> pd.Series:
    sma = source.rolling(window=length, min_periods=length).mean()[:length]
    rest = source[length:]
    return (
        pd.concat([sma, rest])
        .ewm(alpha=1 / length, **kwargs)
        .mean()
    ).rename("RMA")
```

At the start, it seemed that relying solely on .ewm might suffice for RMA calculations. Yet, the results only matched after around 20-30 values. It's clear that if aiming for alignment with TradingView values, these initial discrepancies introduce noticeable inaccuracies. 

To address this, I implemented `_rma_python` to calculate RMA formula via a loop. Initially, I worried this loop could cause lengthy execution times, particularly on lower timeframes like 1 minute. However, after testing it with 1.5 million rows, it proved fast enough, so I kept it. The reason for calling `_rma_pandas` within `_rma_python` is its precise calculation of the initial value, which is always an SMA. This approach helps avoid unnecessary code duplication. Also, I opted for these methods to be private to streamline user function calls. This maintains a resemblance to the TradingView version.

```python
def _rma_python(source: pd.Series, length: int) -> pd.Series:
    alpha = 1 / length
    source_pd = _rma_pandas(source, length)[:length]
    source_values = source[length:].to_numpy().tolist()

    rma_value = float(source_pd.dropna().iloc[0])
    rma_list = [rma_value]

    for source_value in source_values:
        rma_value = alpha * source_value + (1 - alpha) * rma_value
        rma_list.append(rma_value)

    rma_series = pd.Series(
        rma_list,
        name="RMA",
        index=source[length - 1:].index
    )

    return rma_series
```
