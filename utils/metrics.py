# File: utils/metrics.py
"""
Evaluation Metrics - Optimized for speed.
Calculates mAP, Precision, Recall, F1, AP per class.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of boxes - VECTORIZED.
    
    Args:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
        
    Returns:
        IoU matrix (N, M)
    """
    boxes1 = np.expand_dims(boxes1, axis=1)  # (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # (1, M, 4)
    
    # Intersection
    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = boxes1_area + boxes2_area - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Calculate Average Precision using 11-point interpolation."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def calculate_map_fast(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 6
) -> Tuple[float, Dict]:
    """
    Calculate mean Average Precision (mAP) - OPTIMIZED.
    
    Returns:
        (mAP, AP_per_class_dict)
    """
    # Collect predictions and targets per class
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Process predictions
        if len(pred['boxes']) > 0:
            pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
            pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
            pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
            
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                class_predictions[int(label)].append({
                    'image_id': img_idx,
                    'box': box,
                    'score': float(score)
                })
        
        # Process targets
        if len(target['boxes']) > 0:
            target_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
            target_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
            
            for box, label in zip(target_boxes, target_labels):
                class_targets[int(label)].append({
                    'image_id': img_idx,
                    'box': box
                })
    
    # Calculate AP for each class
    ap_per_class = {}
    
    for class_id in range(num_classes):
        preds = class_predictions[class_id]
        gts = class_targets[class_id]
        
        if len(gts) == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        if len(preds) == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        # Sort by score
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        # Match predictions to ground truth
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # Group GT by image
        gt_by_image = defaultdict(list)
        for gt in gts:
            gt_by_image[gt['image_id']].append(gt['box'])
        
        # Track matched GT
        gt_matched = defaultdict(lambda: np.zeros(0, dtype=bool))
        for img_id in gt_by_image:
            gt_matched[img_id] = np.zeros(len(gt_by_image[img_id]), dtype=bool)
        
        # Match each prediction
        for pred_idx, pred in enumerate(preds):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            if img_id not in gt_by_image:
                fp[pred_idx] = 1
                continue
            
            gt_boxes = np.array(gt_by_image[img_id])
            
            # Calculate IoU
            ious = calculate_iou_matrix(
                np.array([pred_box]),
                gt_boxes
            )[0]
            
            # Find best match
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            if max_iou >= iou_threshold:
                if not gt_matched[img_id][max_iou_idx]:
                    tp[pred_idx] = 1
                    gt_matched[img_id][max_iou_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(gts)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Calculate AP
        ap = calculate_ap(recall, precision)
        ap_per_class[class_id] = ap
    
    # Calculate mAP
    mAP = np.mean(list(ap_per_class.values()))
    
    return mAP, ap_per_class


def calculate_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.75],
    num_classes: int = 6
) -> Dict:
    """
    Calculate comprehensive detection metrics.
    
    Returns:
        Dictionary with all metrics including mAP, precision, recall, F1
    """
    metrics = {}
    
    # Calculate mAP at different IoU thresholds
    for iou_thresh in iou_thresholds:
        mAP, ap_per_class = calculate_map_fast(
            predictions, targets, iou_thresh, num_classes
        )
        metrics[f'mAP@{iou_thresh}'] = mAP
        metrics[f'AP_per_class@{iou_thresh}'] = ap_per_class
    
    # Calculate mAP@[0.5:0.95] (COCO-style)
    map_values = []
    for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        mAP, _ = calculate_map_fast(predictions, targets, iou_thresh, num_classes)
        map_values.append(mAP)
    metrics['mAP@[0.5:0.95]'] = np.mean(map_values)
    
    # Calculate precision, recall, F1 per class
    class_metrics = calculate_class_metrics(predictions, targets, num_classes)
    metrics.update(class_metrics)
    
    return metrics


def calculate_class_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    num_classes: int = 6,
    iou_threshold: float = 0.5
) -> Dict:
    """Calculate precision, recall, F1 per class."""
    class_tp = np.zeros(num_classes)
    class_fp = np.zeros(num_classes)
    class_fn = np.zeros(num_classes)
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
        
        target_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
        target_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
        
        if len(pred_boxes) == 0:
            pred_boxes = np.zeros((0, 4))
            pred_labels = np.zeros((0,))
        
        if len(target_boxes) == 0:
            target_boxes = np.zeros((0, 4))
            target_labels = np.zeros((0,))
        
        # Match predictions to targets
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            ious = calculate_iou_matrix(pred_boxes, target_boxes)
            
            matched_targets = set()
            for pred_idx in range(len(pred_boxes)):
                pred_label = int(pred_labels[pred_idx])
                
                best_iou = 0
                best_target_idx = -1
                
                for target_idx in range(len(target_boxes)):
                    if target_idx in matched_targets:
                        continue
                    
                    if int(target_labels[target_idx]) != pred_label:
                        continue
                    
                    if ious[pred_idx, target_idx] > best_iou:
                        best_iou = ious[pred_idx, target_idx]
                        best_target_idx = target_idx
                
                if best_iou >= iou_threshold:
                    class_tp[pred_label] += 1
                    matched_targets.add(best_target_idx)
                else:
                    class_fp[pred_label] += 1
            
            # Count false negatives
            for target_idx in range(len(target_boxes)):
                if target_idx not in matched_targets:
                    class_fn[int(target_labels[target_idx])] += 1
        else:
            # All predictions are FP
            for label in pred_labels:
                class_fp[int(label)] += 1
            
            # All targets are FN
            for label in target_labels:
                class_fn[int(label)] += 1
    
    # Calculate metrics
    precision = class_tp / (class_tp + class_fp + 1e-10)
    recall = class_tp / (class_tp + class_fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1
    }


class MetricsCalculator:
    """Helper class to accumulate predictions and calculate metrics."""
    
    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: List[Dict], targets: List[Dict]):
        """Add batch of predictions and targets."""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> Dict:
        """Compute all metrics."""
        return calculate_metrics(
            self.predictions,
            self.targets,
            iou_thresholds=[0.5, 0.75],
            num_classes=self.num_classes
        )
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []