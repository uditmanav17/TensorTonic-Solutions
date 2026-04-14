from itertools import chain

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    arr = [0] * 256
    for val in chain.from_iterable(image):
        arr[val] += 1
    return arr