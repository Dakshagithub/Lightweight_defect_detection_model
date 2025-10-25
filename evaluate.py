"""
Fixed Evaluation Script for YOLO Format Labels
Replace your existing evaluate.py with this version
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import numpy as np

# Import the YOLO dataset
import sys
sys.path.append(str(Path(__file__).parent))

from models import GhostCAMBWFPN
from datasets.neu_det_yolo import NEUDETDatasetYOLO, collate_fn_yolo
# Note: calculate_map is implemented in the Evaluator class below


class Evaluator:
    """
    Model evaluator with YOLO format labels
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        split: str = 'validation',
        input_size: int = 640,
        batch_size: int = 16,
        num_workers: int = 4,
        device: str = 'cuda',
        conf_thresh: float = 0.001,
        iou_thresh: float = 0.6
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        print("="*70)
        print("Model Evaluation")
        print("="*70)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Data directory: {self.data_dir}")
        print(f"Split: {self.split}")
        print(f"Device: {self.device}")
        print("="*70)
        
        # Load model
        self.load_model()
        
        # Load dataset
        self.load_dataset()
    
    def load_model(self):
        """Load model from checkpoint"""
        print("\nBuilding model...")
        
        # Build model
        self.model = GhostCAMBWFPN(
            num_classes=6,  # NEU-DET has 6 classes
            in_channels=3
        )
        self.model.to(self.device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'ema_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        print("[OK] Model loaded successfully")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Saved metrics: {checkpoint['metrics']}")
    
    def load_dataset(self):
        """Load evaluation dataset"""
        print("\nBuilding dataset...")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            print(f"⚠️  Data directory not found: {self.data_dir}")
            print(f"Current directory: {Path.cwd()}")
            print("\nTrying common paths...")
            
            # Try relative path
            self.data_dir = Path("NEU-DET-bil")
            if not self.data_dir.exists():
                # Try absolute path
                self.data_dir = Path(r"C:\Users\Dhaksha\ghostcam-bwfpn\NEU-DET-bil")
            
            if not self.data_dir.exists():
                raise FileNotFoundError(
                    f"Cannot find data directory. Tried:\n"
                    f"  - {self.data_dir}\n"
                    f"  - NEU-DET-bil\n"
                    f"  - C:\\Users\\Dhaksha\\ghostcam-bwfpn\\NEU-DET-bil"
                )
        
        print(f"✓ Data directory found: {self.data_dir}")
        
        # Create dataset
        self.dataset = NEUDETDatasetYOLO(
            data_dir=str(self.data_dir),
            split=self.split,
            input_size=self.input_size,
            use_mosaic=False,  # No augmentation for evaluation
            mosaic_prob=0.0,
            augment=False
        )
        
        print(f"✓ Dataset: {len(self.dataset)} images")
        print(f"✓ Classes: {self.dataset.class_names}")
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_yolo,
            pin_memory=True
        )
    
    def non_max_suppression(
        self,
        predictions,
        conf_thresh=0.001,
        iou_thresh=0.6,
        max_det=300
    ):
        """
        Apply NMS to predictions
        
        Args:
            predictions: Model outputs 
                        Format: (B, num_anchors, H, W, 4+1+num_classes)
                        or (B, N, 4+1+num_classes)
            conf_thresh: Confidence threshold
            iou_thresh: IoU threshold for NMS
            max_det: Maximum detections per image
        
        Returns:
            List of detections per image, each (N, 6) [x1, y1, x2, y2, conf, cls]
        """
        # Handle different input shapes
        if len(predictions.shape) == 5:
            # Shape: (B, num_anchors, H, W, C) -> (B, H*W*num_anchors, C)
            B, num_anchors, H, W, C = predictions.shape
            predictions = predictions.permute(0, 2, 3, 1, 4)  # (B, H, W, num_anchors, C)
            predictions = predictions.reshape(B, -1, C)  # (B, H*W*num_anchors, C)
        
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[2] - 5
        
        output = []
        
        for i in range(batch_size):
            pred = predictions[i]  # (N, C)
            
            # Extract boxes, objectness, and class scores
            boxes = pred[:, :4]  # (N, 4) [x, y, w, h]
            obj_conf = pred[:, 4]  # (N,)
            class_scores = pred[:, 5:]  # (N, num_classes)
            
            # Get class predictions
            class_conf, class_pred = class_scores.max(dim=1)
            
            # Total confidence
            conf = obj_conf * class_conf
            
            # Filter by confidence
            mask = conf > conf_thresh
            
            if mask.sum() == 0:
                output.append(torch.zeros((0, 6), device=predictions.device))
                continue
            
            boxes = boxes[mask]
            conf = conf[mask]
            class_pred = class_pred[mask]
            
            # Convert from (x, y, w, h) normalized to (x1, y1, x2, y2) pixel coords
            # Scale to image size
            img_size = self.input_size
            
            x_center = boxes[:, 0] * img_size
            y_center = boxes[:, 1] * img_size
            w = boxes[:, 2] * img_size
            h = boxes[:, 3] * img_size
            
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # Clip to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_size)
            
            # NMS per class
            keep = []
            keep_idx = []
            
            for cls in torch.unique(class_pred):
                cls_mask = class_pred == cls
                cls_indices = torch.where(cls_mask)[0]
                cls_boxes = boxes[cls_mask]
                cls_conf = conf[cls_mask]
                
                # Sort by confidence
                sort_idx = torch.argsort(cls_conf, descending=True)
                cls_boxes = cls_boxes[sort_idx]
                cls_conf = cls_conf[sort_idx]
                cls_indices = cls_indices[sort_idx]
                
                # Apply NMS
                keep_mask = torch.ones(len(cls_boxes), dtype=torch.bool, device=predictions.device)
                
                for j in range(len(cls_boxes)):
                    if not keep_mask[j]:
                        continue
                    
                    # Calculate IoU with remaining boxes
                    if j + 1 < len(cls_boxes):
                        iou = self.box_iou(cls_boxes[j:j+1], cls_boxes[j+1:])
                        
                        # Remove overlapping boxes
                        overlap_mask = iou[0] > iou_thresh
                        keep_mask[j+1:] = keep_mask[j+1:] & ~overlap_mask
                
                # Add kept indices
                kept_indices = cls_indices[keep_mask]
                keep_idx.extend(kept_indices.tolist())
            
            if len(keep_idx) > 0:
                keep_idx = torch.tensor(keep_idx, device=predictions.device)
                detections = torch.cat([
                    boxes[keep_idx],
                    conf[keep_idx].unsqueeze(1),
                    class_pred[keep_idx].unsqueeze(1).float()
                ], dim=1)
                
                # Limit detections
                if detections.shape[0] > max_det:
                    detections = detections[:max_det]
                
                output.append(detections.cpu())
            else:
                output.append(torch.zeros((0, 6)))
        
        return output
    
    @staticmethod
    def box_iou(box1, box2):
        """Calculate IoU between boxes"""
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        # Intersection
        x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0])
        y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1])
        x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2])
        y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        union = area1.unsqueeze(1) + area2 - intersection
        
        # IoU
        iou = intersection / (union + 1e-16)
        
        return iou
    
    def calculate_map(self, all_predictions, all_targets, iou_thresholds=[0.5, 0.75]):
        """
        Calculate mAP
        
        Args:
            all_predictions: List of predictions per image (in pixel coords)
            all_targets: List of targets per image (in normalized coords)
            iou_thresholds: IoU thresholds for mAP calculation
        
        Returns:
            mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
        """
        num_classes = 6
        aps = {iou_thresh: [] for iou_thresh in iou_thresholds}
        
        # Count total predictions and targets
        total_preds = sum(len(p) for p in all_predictions)
        total_targets = sum(len(t) for t in all_targets)
        
        print(f"   Total predictions: {total_preds}")
        print(f"   Total ground truth boxes: {total_targets}")
        
        if total_preds == 0:
            print("   ⚠️  No predictions made! Model may not be detecting anything.")
            return 0.0, 0.0, 0.0
        
        for cls in range(num_classes):
            for iou_thresh in iou_thresholds:
                tp = 0
                fp = 0
                fn = 0
                
                for pred, target in zip(all_predictions, all_targets):
                    # Filter by class
                    pred_cls = pred[pred[:, 5] == cls] if len(pred) > 0 else torch.zeros((0, 6))
                    target_cls = target[target[:, 0] == cls] if len(target) > 0 else torch.zeros((0, 5))
                    
                    if len(target_cls) == 0 and len(pred_cls) == 0:
                        continue
                    elif len(target_cls) == 0:
                        fp += len(pred_cls)
                        continue
                    elif len(pred_cls) == 0:
                        fn += len(target_cls)
                        continue
                    
                    # Convert coordinates
                    pred_boxes = pred_cls[:, :4]  # Already in pixel coords from NMS
                    target_boxes = target_cls[:, 1:5]  # Normalized coords
                    
                    # Convert target boxes from normalized to pixel coords
                    img_size = self.input_size
                    target_x_center = target_boxes[:, 0] * img_size
                    target_y_center = target_boxes[:, 1] * img_size
                    target_w = target_boxes[:, 2] * img_size
                    target_h = target_boxes[:, 3] * img_size
                    
                    # Convert to (x1, y1, x2, y2)
                    target_x1 = target_x_center - target_w / 2
                    target_y1 = target_y_center - target_h / 2
                    target_x2 = target_x_center + target_w / 2
                    target_y2 = target_y_center + target_h / 2
                    target_boxes_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)
                    
                    # Calculate IoU
                    iou = self.box_iou(pred_boxes, target_boxes_xyxy)
                    
                    # Match predictions to targets
                    matched = set()
                    for pred_idx in range(len(pred_cls)):
                        best_iou, best_target = iou[pred_idx].max(0)
                        if best_iou >= iou_thresh and best_target.item() not in matched:
                            tp += 1
                            matched.add(best_target.item())
                        else:
                            fp += 1
                    
                    fn += len(target_cls) - len(matched)
                
                # Calculate AP
                if tp + fp > 0 and tp + fn > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    ap = precision * recall
                else:
                    ap = 0.0
                
                aps[iou_thresh].append(ap)
        
        # Calculate mean AP
        map_50 = np.mean(aps[0.5]) if aps[0.5] else 0.0
        map_75 = np.mean(aps[0.75]) if aps[0.75] else 0.0
        map_50_95 = (map_50 + map_75) / 2
        
        # Print per-class AP
        print(f"\n   Per-class AP@0.5:")
        class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        for cls_id, ap in enumerate(aps[0.5]):
            print(f"     {class_names[cls_id]}: {ap:.4f}")
        
        return map_50, map_75, map_50_95
    
    @torch.no_grad()
    def evaluate(self):
        """Run evaluation"""
        print("\n" + "="*70)
        print("Running Evaluation")
        print("="*70)
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.dataloader, desc="Evaluating")
        
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Handle different output formats
            if isinstance(outputs, (list, tuple)):
                # Model returns multiple outputs, use the first one
                predictions = outputs[0] if len(outputs) > 0 else outputs
            else:
                predictions = outputs
            
            # Apply NMS (now handles YOLO format properly)
            detections = self.non_max_suppression(
                predictions,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh
            )
            
            all_predictions.extend(detections)
            
            # Process targets
            batch_targets = []
            for batch_idx in torch.unique(targets[:, 0]):
                target = targets[targets[:, 0] == batch_idx][:, 1:]  # Remove batch index
                batch_targets.append(target.cpu())
            all_targets.extend(batch_targets)
        
        # Calculate mAP
        print("\nCalculating mAP...")
        map_50, map_75, map_50_95 = self.calculate_map(all_predictions, all_targets)
        
        # Print results
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)
        print(f"mAP@0.5:       {map_50:.4f}")
        print(f"mAP@0.75:      {map_75:.4f}")
        print(f"mAP@[0.5:0.95]: {map_50_95:.4f}")
        print("="*70)
        
        return {
            'mAP@0.5': map_50,
            'mAP@0.75': map_75,
            'mAP@[0.5:0.95]': map_50_95
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate GhostCAM-BWFPN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, default='NEU-DET-bil',
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--conf_thresh', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.6,
                        help='IoU threshold for NMS')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh
    )
    
    # Run evaluation
    results = evaluator.evaluate()


if __name__ == '__main__':
    main()