# File: utils/postprocessing.py
"""
Post-processing filters - PHASE 1.
Aggressive filtering to eliminate false positives.
"""

import torch
from typing import Tuple
from .cnms import confluence_nms


def filter_by_box_size(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_size: int = 15,
    max_size: int = 190,
    min_area: int = 300
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter boxes by size - removes tiny false positive boxes.
    
    Why: Low precision suggests many small spurious detections.
    Steel defects have minimum realistic sizes.
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    
    size_mask = (
        (widths >= min_size) & (widths <= max_size) &
        (heights >= min_size) & (heights <= max_size) &
        (areas >= min_area)
    )
    
    return boxes[size_mask], scores[size_mask], labels[size_mask]


def filter_by_aspect_ratio(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter by aspect ratio - removes unrealistic box shapes.
    
    Why: Steel defects have realistic shape constraints.
    Very elongated boxes are likely false positives.
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ratios = widths / (heights + 1e-6)
    
    ratio_mask = (ratios >= min_ratio) & (ratios <= max_ratio)
    
    return boxes[ratio_mask], scores[ratio_mask], labels[ratio_mask]


def filter_by_edge_distance(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_size: int = 200,
    min_distance: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter boxes too close to image edges.
    
    Why: Boxes at edges are often artifacts from padding/border effects.
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    edge_mask = (
        (boxes[:, 0] >= min_distance) &
        (boxes[:, 1] >= min_distance) &
        (boxes[:, 2] <= image_size - min_distance) &
        (boxes[:, 3] <= image_size - min_distance)
    )
    
    return boxes[edge_mask], scores[edge_mask], labels[edge_mask]


def multi_stage_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    confluence_threshold: float = 0.60,
    score_threshold: float = 0.35
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-stage NMS: C-NMS + confidence filtering.
    
    Why: Single-stage NMS is insufficient. Need aggressive suppression.
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Apply C-NMS
    keep_indices = confluence_nms(
        boxes=boxes,
        scores=scores,
        labels=labels,
        confluence_threshold=confluence_threshold,
        score_threshold=score_threshold
    )
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]