def box_area(x1, y1, x2, y2):
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height

def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    inter_box = [
        max(box_a[0], box_b[0]), 
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    ]
    intersection_area = box_area(*inter_box)
    union = box_area(*box_a) + box_area(*box_b) - intersection_area
    # print(intersection_area, union)
    return intersection_area / union if union != 0 else 0
    