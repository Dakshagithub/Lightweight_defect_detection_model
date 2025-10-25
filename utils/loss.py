# File: utils/loss.py
"""
Loss Functions with EIoU and Focal Loss (Phase 2 & 4).
Includes all optimizations for reducing false positives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


def compute_eiou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute Efficient IoU (EIoU) loss - PHASE 4.
    
    EIoU improves upon GIoU by penalizing:
    1. Center point distance
    2. Width difference
    3. Height difference
    
    This significantly improves localization accuracy.
    
    Args:
        box1: Boxes of shape (N, 4) in format [x1, y1, x2, y2]
        box2: Boxes of shape (N, 4) in format [x1, y1, x2, y2]
        
    Returns:
        EIoU values of shape (N,)
    """
    # Intersection
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Center points
    box1_cx = (box1[:, 0] + box1[:, 2]) / 2
    box1_cy = (box1[:, 1] + box1[:, 3]) / 2
    box2_cx = (box2[:, 0] + box2[:, 2]) / 2
    box2_cy = (box2[:, 1] + box2[:, 3]) / 2
    
    # Smallest enclosing box
    enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
    enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
    enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
    enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
    
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_c2 = enclose_w ** 2 + enclose_h ** 2 + 1e-7
    
    # Center distance penalty
    rho2 = (box1_cx - box2_cx) ** 2 + (box1_cy - box2_cy) ** 2
    center_penalty = rho2 / enclose_c2
    
    # Width and height penalties
    box1_w = box1[:, 2] - box1[:, 0]
    box1_h = box1[:, 3] - box1[:, 1]
    box2_w = box2[:, 2] - box2[:, 0]
    box2_h = box2[:, 3] - box2[:, 1]
    
    cw2 = enclose_w ** 2 + 1e-7
    ch2 = enclose_h ** 2 + 1e-7
    
    width_penalty = (box1_w - box2_w) ** 2 / cw2
    height_penalty = (box1_h - box2_h) ** 2 / ch2
    
    # EIoU
    eiou = iou - center_penalty - width_penalty - height_penalty
    
    return eiou


def compute_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (backup function).
    """
    # Intersection
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Smallest enclosing box
    enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
    enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
    enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
    enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
    
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_area = enclose_w * enclose_h
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
    return giou


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing hard examples - PHASE 2.
    
    Focal Loss down-weights easy examples and focuses training
    on hard negatives. This is critical for reducing false positives.
    
    FL(pt) = -α(1-pt)^γ * log(pt)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits (B, ...)
            targets: Binary targets (B, ...)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class YOLOLoss(nn.Module):
    """
    YOLO Loss with all optimizations:
    - Phase 2: Higher objectness weight + Focal Loss
    - Phase 4: EIoU loss for better localization
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        img_size: int = 200,
        lambda_box: float = 0.05,
        lambda_obj: float = 15.0,  # PHASE 2: Increased from 1.0
        lambda_cls: float = 0.5,
        label_smoothing: float = 0.005
    ):
        super(YOLOLoss, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Loss weights
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj * (img_size / 416) ** 2
        self.lambda_cls = lambda_cls * (num_classes / 80)
        
        self.label_smoothing = label_smoothing
        
        # PHASE 2: Focal Loss for objectness
        self.focal_obj = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[Dict],
        model: nn.Module
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss with all optimizations.
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        
        anchors = model.head.anchors
        
        for scale_idx, pred in enumerate(predictions):
            b, na, h, w, no = pred.shape
            stride = self.img_size // h
            
            # Build targets
            tbox, tobj, tcls, indices = self._build_targets(
                pred, targets, anchors[scale_idx], stride, h, w
            )
            
            if len(indices[0]) == 0:
                # No targets - only objectness loss
                obj_pred = pred[..., 4]
                loss_obj += self.focal_obj(obj_pred, tobj)
                continue
            
            # Extract predictions for matched anchors
            b_idx, a_idx, gi, gj = indices
            pred_matched = pred[b_idx, a_idx, gj, gi]
            
            # PHASE 4: EIoU Loss for boxes
            pred_xy = pred_matched[:, :2].sigmoid()
            pred_wh = pred_matched[:, 2:4].exp() * torch.tensor(
                anchors[scale_idx], device=device
            ).float()[a_idx] / stride
            
            # Grid positions
            grid_xy = torch.stack([gi.float(), gj.float()], dim=1)
            
            # Convert to absolute coordinates in grid space
            pred_xy_abs = pred_xy + grid_xy
            pred_boxes_grid = torch.cat([
                pred_xy_abs - pred_wh / 2,
                pred_xy_abs + pred_wh / 2
            ], dim=1)
            
            target_boxes_grid = tbox
            
            # EIoU loss
            eiou = compute_eiou(pred_boxes_grid, target_boxes_grid)
            loss_box += (1.0 - eiou).mean()
            
            # PHASE 2: Focal Loss for objectness + background penalty
            obj_pred = pred[..., 4]
            tobj[b_idx, a_idx, gj, gi] = eiou.detach().clamp(0)
            
            # Focal loss
            loss_obj += self.focal_obj(obj_pred, tobj)
            
            # Additional background penalty
            background_mask = (tobj == 0)
            if background_mask.any():
                background_conf = torch.sigmoid(obj_pred[background_mask])
                false_positive_penalty = (background_conf ** 2).mean() * 0.5
                loss_obj += false_positive_penalty
            
            # Classification loss
            if self.num_classes > 1:
                cls_pred = pred_matched[:, 5:]
                
                # Label smoothing
                tcls_smooth = torch.full_like(cls_pred, self.label_smoothing)
                tcls_smooth[range(len(tcls)), tcls] = 1.0 - self.label_smoothing
                
                loss_cls += self.bce_cls(cls_pred, tcls_smooth)
        
        # Total loss
        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )
        
        loss_dict = {
            'loss': total_loss.item(),
            'loss_box': loss_box.item(),
            'loss_obj': loss_obj.item(),
            'loss_cls': loss_cls.item()
        }
        
        return total_loss, loss_dict
    
    def _build_targets(
        self,
        pred: torch.Tensor,
        targets: List[Dict],
        anchors: List[Tuple],
        stride: int,
        grid_h: int,
        grid_w: int
    ) -> Tuple:
        """
        Build targets for a specific scale with improved anchor matching.
        """
        batch_size, na, _, _, _ = pred.shape
        device = pred.device
        
        tobj = torch.zeros(batch_size, na, grid_h, grid_w, device=device)
        tbox_list = []
        tcls_list = []
        indices_list = []
        
        anchor_tensor = torch.tensor(anchors, device=device).float()
        
        for batch_idx, target in enumerate(targets):
            if len(target['boxes']) == 0:
                continue
            
            boxes = target['boxes'].to(device)
            labels = target['labels'].to(device)
            
            # Convert boxes to grid coordinates
            boxes_grid = boxes.clone()
            boxes_grid[:, [0, 2]] /= stride
            boxes_grid[:, [1, 3]] /= stride
            
            # Get box centers and sizes
            box_xy = (boxes_grid[:, :2] + boxes_grid[:, 2:]) / 2
            box_wh = boxes_grid[:, 2:] - boxes_grid[:, :2]
            
            # Scale anchors to grid
            anchor_wh = anchor_tensor / stride
            
            for obj_idx in range(len(boxes)):
                gx, gy = box_xy[obj_idx]
                gi, gj = int(gx), int(gy)
                
                # Check bounds
                if not (0 <= gi < grid_w and 0 <= gj < grid_h):
                    continue
                
                # Find best matching anchor
                box_wh_obj = box_wh[obj_idx]
                ratio = box_wh_obj.unsqueeze(0) / anchor_wh
                ratio = torch.max(ratio, 1 / ratio).max(dim=1)[0]
                
                # Use anchors with ratio < 4.0
                anchor_indices = (ratio < 4.0).nonzero(as_tuple=False).squeeze(1)
                
                if len(anchor_indices) == 0:
                    anchor_indices = ratio.argmin().unsqueeze(0)
                
                for anchor_idx in anchor_indices:
                    indices_list.append((
                        batch_idx,
                        anchor_idx.item(),
                        gi,
                        gj
                    ))
                    
                    tbox_list.append(boxes_grid[obj_idx])
                    tcls_list.append(labels[obj_idx])
                    tobj[batch_idx, anchor_idx, gj, gi] = 1.0
        
        if len(indices_list) > 0:
            indices = tuple(zip(*indices_list))
            indices = (
                torch.tensor(indices[0], device=device, dtype=torch.long),
                torch.tensor(indices[1], device=device, dtype=torch.long),
                torch.tensor(indices[2], device=device, dtype=torch.long),
                torch.tensor(indices[3], device=device, dtype=torch.long)
            )
            tbox = torch.stack(tbox_list).to(device)
            tcls = torch.stack(tcls_list).to(device)
        else:
            indices = (torch.tensor([], device=device, dtype=torch.long),) * 4
            tbox = torch.zeros((0, 4), device=device)
            tcls = torch.zeros((0,), device=device, dtype=torch.long)
        
        return tbox, tobj, tcls, indices