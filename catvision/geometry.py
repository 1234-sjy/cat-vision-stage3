from __future__ import annotations

import math

import numpy as np


def crop_box(box, size: tuple[int, int], margin: float = 0.06) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid subject box")
    width, height = size
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    result = (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
              min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Subject box is outside the image")
    return result


def main_cat_index(boxes, class_ids, cat_id: int = 15) -> int | None:
    candidates = [(max(0, b[2] - b[0]) * max(0, b[3] - b[1]), i)
                  for i, (b, c) in enumerate(zip(boxes, class_ids)) if int(c) == cat_id]
    return max(candidates)[1] if candidates else None


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
