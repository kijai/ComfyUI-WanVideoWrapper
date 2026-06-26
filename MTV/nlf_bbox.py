"""NLF bbox formatting helpers."""

from __future__ import annotations


def format_nlf_detected_boxes(all_boxes):
    """Format detector xyxy rows for the public BBOX output."""

    formatted_boxes = []
    for box in all_boxes:
        if hasattr(box, "detach"):
            box = box.detach()
        if hasattr(box, "cpu"):
            box = box.cpu()
        if hasattr(box, "numel") and box.numel() == 0:
            formatted_boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        rows = box.tolist() if hasattr(box, "tolist") else box
        if not rows:
            formatted_boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        if isinstance(rows[0], (bool, int, float)):
            rows = [rows]
        candidates = []
        for row in rows:
            if len(row) < 4:
                continue
            x_min, y_min, x_max, y_max = (float(row[index]) for index in range(4))
            if x_max <= x_min or y_max <= y_min:
                continue
            candidates.append([x_min, y_min, x_max, y_max])
        if not candidates:
            formatted_boxes.append([0.0, 0.0, 0.0, 0.0])
        elif len(candidates) == 1:
            formatted_boxes.append(candidates[0])
        else:
            formatted_boxes.append(candidates)
    return formatted_boxes
