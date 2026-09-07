def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    if n_items == 0:
        return 0
    
    recommended_items = set().union(*recommendations)

    return len(recommended_items) / n_items