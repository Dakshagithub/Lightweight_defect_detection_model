# File: models/detection_head.py
"""
YOLOv5-style Detection Head with optimized anchors.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class DetectionHead(nn.Module):
    """
    YOLOv5-style detection head with 3 detection scales.
    Uses OPTIMIZED anchors for NEU-DET dataset.
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        anchors: List[List[Tuple[int, int]]] = None
    ):
        super(DetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes
        self.num_anchors = 3
        
        # Anchors will be optimized - placeholder for now
        if anchors is None:
            # These will be replaced by optimized anchors
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)]
            ]
        else:
            self.anchors = anchors
        
        # Detection convolutions
        in_channels = 128
        out_channels = self.num_anchors * self.num_outputs
        
        self.detect_p3 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.detect_p4 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.detect_p5 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
        # Register anchors as buffers
        for i, anchor_list in enumerate(self.anchors):
            self.register_buffer(
                f'anchor_grid_{i}',
                torch.tensor(anchor_list).float().view(1, self.num_anchors, 1, 1, 2)
            )
    
    def forward(self, p3, p4, p5):
        """Forward pass."""
        batch_size = p3.shape[0]
        
        # P3 detection
        pred_p3 = self.detect_p3(p3)
        pred_p3 = pred_p3.view(
            batch_size, self.num_anchors, self.num_outputs,
            pred_p3.shape[2], pred_p3.shape[3]
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        # P4 detection
        pred_p4 = self.detect_p4(p4)
        pred_p4 = pred_p4.view(
            batch_size, self.num_anchors, self.num_outputs,
            pred_p4.shape[2], pred_p4.shape[3]
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        # P5 detection
        pred_p5 = self.detect_p5(p5)
        pred_p5 = pred_p5.view(
            batch_size, self.num_anchors, self.num_outputs,
            pred_p5.shape[2], pred_p5.shape[3]
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        return pred_p3, pred_p4, pred_p5
    
    def decode_predictions(
        self,
        predictions: List[torch.Tensor],
        input_size: int = 200,
        conf_threshold: float = 0.001
    ) -> List[torch.Tensor]:
        """Decode predictions to bounding boxes."""
        batch_size = predictions[0].shape[0]
        all_predictions = []
        
        for batch_idx in range(batch_size):
            batch_preds = []
            
            for scale_idx, pred in enumerate(predictions):
                b, na, h, w, no = pred.shape
                stride = input_size // h
                
                pred_img = pred[batch_idx].reshape(-1, no)
                
                device = pred_img.device
                
                # Create grid
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing='ij'
                )
                
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
                grid_xy = grid_xy.unsqueeze(0).repeat(na, 1, 1, 1)
                grid_xy = grid_xy.reshape(-1, 2)
                
                # Anchors
                anchors_tensor = torch.tensor(
                    self.anchors[scale_idx],
                    device=device,
                    dtype=torch.float32
                )
                anchors_tensor = anchors_tensor.unsqueeze(1).unsqueeze(1)
                anchors_tensor = anchors_tensor.repeat(1, h, w, 1)
                anchors_tensor = anchors_tensor.reshape(-1, 2)
                
                # Decode
                xy = (torch.sigmoid(pred_img[:, :2]) + grid_xy) * stride
                wh = torch.exp(pred_img[:, 2:4]) * anchors_tensor
                conf = torch.sigmoid(pred_img[:, 4:5])
                cls_scores = torch.sigmoid(pred_img[:, 5:])
                
                # Combined confidence
                cls_conf = conf * cls_scores
                max_cls_conf, _ = cls_conf.max(dim=1, keepdim=True)
                
                # Convert to x1y1x2y2
                x1y1 = xy - wh / 2
                x2y2 = xy + wh / 2
                boxes = torch.cat([x1y1, x2y2], dim=1)
                
                decoded = torch.cat([boxes, max_cls_conf, cls_conf], dim=1)
                
                mask = max_cls_conf.squeeze() >= conf_threshold
                decoded = decoded[mask]
                
                batch_preds.append(decoded)
            
            if len(batch_preds) > 0:
                all_preds = torch.cat(batch_preds, dim=0)
            else:
                all_preds = torch.zeros((0, 4 + 1 + self.num_classes), device=predictions[0].device)
            
            all_predictions.append(all_preds)
        
        return all_predictions