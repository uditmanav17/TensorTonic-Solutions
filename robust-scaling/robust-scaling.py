def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    # Convert to a sorted float array
    arr = np.asarray(values, dtype=np.float64)
    s = np.sort(arr)
    n = len(s)
    if n == 1:
        return [0]
    
    # Find the median and split into upper/lower halves
    mid_idx = n // 2
    if n % 2 == 1:
        # Odd length: Exclude the exact middle element from both halves
        median = s[mid_idx]
        lower_half = s[:mid_idx]
        upper_half = s[mid_idx + 1:]
    else:
        # Even length: Split right down the middle
        median = (s[mid_idx - 1] + s[mid_idx]) / 2.0
        lower_half = s[:mid_idx]
        upper_half = s[mid_idx:]
    
    # Calculate Q1 and Q3 as the medians of the respective halves
    q1 = np.median(lower_half)
    q3 = np.median(upper_half)
    
    deno = q3 - q1
    
    # Fallback to prevent division by zero if all values are identical
    if deno == 0:
        deno = 1.0
        
    return (arr - median) / deno
