def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox

    return int((x1 + x1) / 2), int((y1 + y1) / 2)

def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])