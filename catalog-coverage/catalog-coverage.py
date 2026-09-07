def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    # Write code here
    if n_items == 0:
        return 0
    
    items = set()
    for li in recommendations:
        items = items.union(li)

    return len(items) / n_items
