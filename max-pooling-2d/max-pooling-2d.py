def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    rows = len(X)
    cols = len(X[0])

    result = []

    for i in range(0, rows - pool_size + 1, pool_size):
        row = []

        for j in range(0, cols - pool_size + 1, pool_size):
            window_max = max(
                X[r][c]
                for r in range(i, i + pool_size)
                for c in range(j, j + pool_size)
            )

            row.append(window_max)

        result.append(row)

    return result
