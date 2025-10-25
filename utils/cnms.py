# File: utils/cnms.py
"""
Confluence Non-Maximum Suppression (C-NMS) - PHASE 1.
Based on IEEE TPAMI 2023 paper.
"""

import torch
import numpy as np
from typing import Tuple, List


def normalize_coordinates(boxes: torch.Tensor) -> torch.Tensor:
    """Normalize box coordinates to [0, 1] for scale-invariance."""
    if len(boxes) == 0:
        return boxes
    
    x_coords = boxes[:, [0, 2]].flatten()
    y_coords = boxes[:, [1, 3]].flatten()
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_range = x_max - x_min + 1e-6
    y_range = y_max - y_min + 1e-6
    
    boxes_norm = boxes.clone()
    boxes_norm[:, [0, 2]] = (boxes[:, [0, 2]] - x_min) / x_range
    boxes_norm[:, [1, 3]] = (boxes[:, [1, 3]] - y_min) / y_range
    
    return boxes_norm


def compute_proximity(box_i: torch.Tensor, box_j: torch.Tensor) -> float:
    """
    Compute normalized Manhattan distance (proximity).
    P(bi, bj) = |xu_j - xu_i| + |yu_j - yu_i| + |xv_j - xv_i| + |yv_j - yv_i|
    """
    proximity = (
        abs(box_j[0] - box_i[0]) +
        abs(box_j[1] - box_i[1]) +
        abs(box_j[2] - box_i[2]) +
        abs(box_j[3] - box_i[3])
    )
    return proximity.item()


def compute_pairwise_proximity_matrix(boxes_norm: torch.Tensor) -> torch.Tensor:
    """Compute pairwise proximity matrix for all boxes."""
    N = len(boxes_norm)
    proximity_matrix = torch.zeros((N, N), device=boxes_norm.device)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                proximity_matrix[i, j] = compute_proximity(boxes_norm[i], boxes_norm[j])
    
    return proximity_matrix


def compute_mean_proximity(
    box_idx: int,
    cluster_indices: List[int],
    proximity_matrix: torch.Tensor
) -> float:
    """Compute mean proximity for a box within its cluster."""
    if len(cluster_indices) == 0:
        return 0.0
    
    proximities = [proximity_matrix[box_idx, j].item() 
                   for j in cluster_indices if j != box_idx]
    
    if len(proximities) == 0:
        return 0.0
    
    return sum(proximities) / len(proximities)


def compute_weighted_proximity(mean_proximity: float, confidence: float) -> float:
    """
    Compute confidence-weighted proximity.
    Pw(bi) = P(bi) × (1 - si)
    """
    return mean_proximity * (1.0 - confidence)


def confluence_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor = None,
    confluence_threshold: float = 0.65,  # PHASE 1: Aggressive
    score_threshold: float = 0.35
) -> torch.Tensor:
    """
    Confluence Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        labels: (N,) - optional for per-class NMS
        confluence_threshold: Lower = more aggressive suppression
        score_threshold: Minimum confidence
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Filter by score
    score_mask = scores >= score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    if labels is not None:
        labels = labels[score_mask]
    
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Per-class NMS
    if labels is not None:
        keep_indices = []
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            label_mask = labels == label
            label_boxes = boxes[label_mask]
            label_scores = scores[label_mask]
            label_indices = torch.where(label_mask)[0]
            
            keep_label = confluence_nms_single_class(
                label_boxes,
                label_scores,
                confluence_threshold
            )
            
            keep_indices.extend(label_indices[keep_label].tolist())
        
        return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
    else:
        return confluence_nms_single_class(boxes, scores, confluence_threshold)


def confluence_nms_single_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    confluence_threshold: float
) -> torch.Tensor:
    """C-NMS for single class."""
    N = len(boxes)
    
    if N == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Normalize coordinates
    boxes_norm = normalize_coordinates(boxes)
    
    # Compute pairwise proximity
    proximity_matrix = compute_pairwise_proximity_matrix(boxes_norm)
    
    # Clustering and suppression
    suppressed = torch.zeros(N, dtype=torch.bool, device=boxes.device)
    keep_indices = []
    
    # Sort by confidence
    sorted_indices = torch.argsort(scores, descending=True)
    
    for idx in sorted_indices:
        if suppressed[idx]:
            continue
        
        # Find cluster
        cluster_mask = proximity_matrix[idx] < confluence_threshold
        cluster_indices = torch.where(cluster_mask)[0].tolist()
        
        if len(cluster_indices) == 0:
            keep_indices.append(idx.item())
            continue
        
        cluster_indices.append(idx.item())
        
        # Find best box in cluster
        best_idx = idx.item()
        best_pw = float('inf')
        
        for cluster_idx in cluster_indices:
            if suppressed[cluster_idx]:
                continue
            
            other_indices = [j for j in cluster_indices if j != cluster_idx]
            mean_prox = compute_mean_proximity(cluster_idx, other_indices, proximity_matrix)
            pw = compute_weighted_proximity(mean_prox, scores[cluster_idx].item())
            
            if pw < best_pw:
                best_pw = pw
                best_idx = cluster_idx
        
        # Keep best, suppress others
        keep_indices.append(best_idx)
        for cluster_idx in cluster_indices:
            if cluster_idx != best_idx:
                suppressed[cluster_idx] = True
    
    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)


def cnms(boxes: torch.Tensor, scores: torch.Tensor, confluence_threshold: float = 0.65) -> torch.Tensor:
    """Simplified C-NMS API."""
    return confluence_nms(boxes, scores, confluence_threshold=confluence_threshold)