def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    output = []
    val = 1.0
    for r in returns:
        final_return = 1 + r
        val = val * final_return
        output.append(val - 1)
    return output