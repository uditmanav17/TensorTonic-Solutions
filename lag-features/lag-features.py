def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    # Write code here
    ans = []
    for idx, val in enumerate(series):
        temp = []
        if idx < max(lags): continue
        for idx2 in lags:
            if idx - idx2 >= 0:
                temp.append(series[idx - idx2])
            else:
                temp.append(None)
        ans.append(temp)
    return ans
