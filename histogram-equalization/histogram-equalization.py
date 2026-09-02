import numpy as np

def histogram_equalize(image: list) -> list:
    """
    Returns the histogram-equalized grayscale image.
    """

    image = np.asarray(image)

    # 1. Histogram: intensity values 0..255
    hist = np.bincount(image.ravel(), minlength=256)

    # 2. CDF
    cdf = np.cumsum(hist)

    # 3. Smallest non-zero CDF
    cdf_min = cdf[cdf > 0][0]

    total_pixels = image.size

    # 4. Handle image where all pixels have the same value
    if total_pixels == cdf_min:
        return np.zeros_like(image).tolist()

    # Create mapping for 0..255
    mapping = np.round(
        (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    ).astype(np.uint8)

    # Apply mapping
    return mapping[image].tolist()
