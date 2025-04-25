def get_bbox_center(bbox):
    """
    Calculate the center coordinates of a bounding box.
    
    Args:
        bbox (list or tuple): Bounding box coordinates in format [x1, y1, x2, y2] where:
            - x1, y1: Top-left corner coordinates
            - x2, y2: Bottom-right corner coordinates
    
    Returns:
        tuple: (center_x, center_y) coordinates as integers
    """
    x1, y1, x2, y2 = bbox
    
    # Note: There appears to be a bug here - should use x1+x2 and y1+y2 instead of x1+x1 and y1+y1
    return int((x1 + x1) / 2), int((y1 + y1) / 2)  # Bug: Should be (x1+x2)/2, (y1+y2)/2

def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.
    
    Args:
        bbox (list or tuple): Bounding box coordinates in format [x1, y1, x2, y2] where:
            - x1: Left edge x-coordinate
            - x2: Right edge x-coordinate
    
    Returns:
        int: Width of the bounding box in pixels
    """
    return int(bbox[2] - bbox[0])  # Width = right edge (x2) - left edge (x1)