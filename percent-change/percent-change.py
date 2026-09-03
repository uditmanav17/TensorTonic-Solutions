def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    N = len(series)
    ans = []
    for idx in range(1, N):
        nume = series[idx] - series[idx - 1]
        deno = series[idx-1]
        ans.append(0 if deno == 0 else (nume / deno))
    return ans
