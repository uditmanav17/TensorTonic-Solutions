import numpy as np

def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    image = np.array(image, dtype=np.float64)
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray.tolist()
