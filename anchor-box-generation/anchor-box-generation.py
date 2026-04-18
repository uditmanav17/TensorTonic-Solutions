def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    anchors = []
    stride = image_size / feature_size
    
    # 1. Iterate over grid cells (row-major: i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            # 2. Compute center of the current cell
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # 3. For each cell, iterate over scales then aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    # 4. Calculate width and height based on ratio
                    # w = s * sqrt(r), h = s / sqrt(r)
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # 5. Define box as [x1, y1, x2, y2]
                    anchor = [
                        cx - w / 2, # x1
                        cy - h / 2, # y1
                        cx + w / 2, # x2
                        cy + h / 2  # y2
                    ]
                    anchors.append(anchor)
                    
    return anchors