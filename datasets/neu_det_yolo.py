"""
Updated NEU-DET Dataset with YOLO Format Labels and Mosaic Augmentation

Place this file at: datasets/neu_det_yolo.py
Or replace the relevant parts in your existing datasets/__init__.py
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NEUDETDatasetYOLO(torch.utils.data.Dataset):
    """
    NEU-DET Dataset with YOLO format labels and mosaic augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        input_size: int = 640,
        use_mosaic: bool = True,
        mosaic_prob: float = 0.5,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Base directory (e.g., 'NEU-DET-bil')
            split: 'train' or 'validation'
            input_size: Target image size
            use_mosaic: Whether to use mosaic augmentation
            mosaic_prob: Probability of applying mosaic (0-1)
            augment: Whether to apply other augmentations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_size = input_size
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.augment = augment
        self.current_epoch = 0
        self.mosaic_epochs = 70  # Use mosaic for first 70 epochs
        
        # Class names
        self.class_names = [
            'crazing',
            'inclusion',
            'patches',
            'pitted_surface',
            'rolled-in_scale',
            'scratches'
        ]
        self.num_classes = len(self.class_names)
        
        # Paths
        self.label_dir = self.data_dir / split / 'labels'
        self.image_dir = self.data_dir / split / 'images'
        
        # Get all label files
        self.label_files = sorted(list(self.label_dir.glob('*.txt')))
        
        # Filter out empty labels if needed
        self.label_files = [f for f in self.label_files if f.stat().st_size > 0]
        
        if len(self.label_files) == 0:
            raise ValueError(f"No label files found in {self.label_dir}")
        
        print(f"Loaded {len(self.label_files)} samples for {split}")
        
        # Standard augmentations (non-mosaic)
        if augment and split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.2),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
                min_area=100
            ))
        else:
            self.transform = None
    
    def set_epoch(self, epoch: int):
        """Update current epoch for mosaic control"""
        self.current_epoch = epoch
    
    def should_use_mosaic(self) -> bool:
        """Check if mosaic should be applied in current epoch"""
        if not self.use_mosaic or self.split != 'train':
            return False
        return self.current_epoch < self.mosaic_epochs and random.random() < self.mosaic_prob
    
    def find_image_path(self, label_path: Path) -> Optional[Path]:
        """Find corresponding image for a label file"""
        label_name = label_path.stem
        
        # Try to find in class subdirectories
        for ext in ['.jpg', '.png', '.jpeg']:
            # Search in all subdirectories
            for img_path in self.image_dir.rglob(label_name + ext):
                return img_path
        
        return None
    
    def load_image_and_labels(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and labels for a given index"""
        label_path = self.label_files[idx]
        img_path = self.find_image_path(label_path)
        
        if img_path is None or not img_path.exists():
            # Return black image and empty labels
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8), np.zeros((0, 5))
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8), np.zeros((0, 5))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels (already in YOLO format)
        labels = np.loadtxt(label_path, ndmin=2)
        
        return img, labels
    
    def apply_mosaic(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation to 4 images
        
        Args:
            indices: List of 4 indices
        
        Returns:
            mosaic_img: Combined image
            mosaic_labels: Combined labels
        """
        # Create mosaic canvas
        mosaic_img = np.full((self.input_size * 2, self.input_size * 2, 3), 114, dtype=np.uint8)
        
        # Random center point
        yc = int(random.uniform(0.5 * self.input_size, 1.5 * self.input_size))
        xc = int(random.uniform(0.5 * self.input_size, 1.5 * self.input_size))
        
        mosaic_labels = []
        
        for i, idx in enumerate(indices):
            img, labels = self.load_image_and_labels(idx)
            h, w = img.shape[:2]
            
            # Resize to input size
            if h != self.input_size or w != self.input_size:
                img = cv2.resize(img, (self.input_size, self.input_size))
            
            # Determine position in mosaic
            if i == 0:  # Top-left
                x1a, y1a = max(xc - self.input_size, 0), max(yc - self.input_size, 0)
                x2a, y2a = xc, yc
                x1b, y1b = self.input_size - (x2a - x1a), self.input_size - (y2a - y1a)
                x2b, y2b = self.input_size, self.input_size
            elif i == 1:  # Top-right
                x1a, y1a = xc, max(yc - self.input_size, 0)
                x2a, y2a = min(xc + self.input_size, self.input_size * 2), yc
                x1b, y1b = 0, self.input_size - (y2a - y1a)
                x2b, y2b = min(self.input_size, x2a - x1a), self.input_size
            elif i == 2:  # Bottom-left
                x1a, y1a = max(xc - self.input_size, 0), yc
                x2a, y2a = xc, min(self.input_size * 2, yc + self.input_size)
                x1b, y1b = self.input_size - (x2a - x1a), 0
                x2b, y2b = self.input_size, min(y2a - y1a, self.input_size)
            else:  # Bottom-right
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + self.input_size, self.input_size * 2), min(self.input_size * 2, yc + self.input_size)
                x1b, y1b = 0, 0
                x2b, y2b = min(self.input_size, x2a - x1a), min(y2a - y1a, self.input_size)
            
            # Place image
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust labels
            if labels is not None and len(labels) > 0:
                labels_copy = labels.copy()
                # Convert normalized to pixel coordinates
                labels_copy[:, 1] = self.input_size * labels[:, 1] + (x1a - x1b)
                labels_copy[:, 2] = self.input_size * labels[:, 2] + (y1a - y1b)
                labels_copy[:, 3] = self.input_size * labels[:, 3]
                labels_copy[:, 4] = self.input_size * labels[:, 4]
                mosaic_labels.append(labels_copy)
        
        # Combine labels
        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            
            # Clip to valid range
            mosaic_labels[:, 1:] = np.clip(mosaic_labels[:, 1:], 0, 2 * self.input_size)
            
            # Convert back to normalized coordinates
            mosaic_labels[:, 1] /= (2 * self.input_size)
            mosaic_labels[:, 2] /= (2 * self.input_size)
            mosaic_labels[:, 3] /= (2 * self.input_size)
            mosaic_labels[:, 4] /= (2 * self.input_size)
            
            # Filter valid boxes
            valid = (mosaic_labels[:, 3] > 0.01) & (mosaic_labels[:, 4] > 0.01)
            mosaic_labels = mosaic_labels[valid]
        else:
            mosaic_labels = np.zeros((0, 5))
        
        # Resize to input size
        mosaic_img = cv2.resize(mosaic_img, (self.input_size, self.input_size))
        
        return mosaic_img, mosaic_labels
    
    def __len__(self) -> int:
        return len(self.label_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get item with optional mosaic augmentation
        
        Returns:
            image: Tensor of shape (3, H, W), normalized to [0, 1]
            labels: Array of shape (N, 5) [class_id, x_center, y_center, w, h]
        """
        # Check if we should use mosaic
        if self.should_use_mosaic():
            # Get 4 random images for mosaic
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            img, labels = self.apply_mosaic(indices)
        else:
            # Load single image
            img, labels = self.load_image_and_labels(idx)
            # Resize to input size
            img = cv2.resize(img, (self.input_size, self.input_size))
        
        # Apply standard augmentations
        if self.transform is not None and len(labels) > 0:
            class_labels = labels[:, 0].astype(int).tolist()
            bboxes = labels[:, 1:].tolist()
            
            try:
                transformed = self.transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                img = transformed['image']
                
                # Reconstruct labels
                if len(transformed['bboxes']) > 0:
                    bboxes = np.array(transformed['bboxes'])
                    class_labels = np.array(transformed['class_labels']).reshape(-1, 1)
                    labels = np.concatenate([class_labels, bboxes], axis=1)
                else:
                    labels = np.zeros((0, 5))
            except Exception as e:
                # If augmentation fails, use original
                print(f"Augmentation failed: {e}")
        
        # Normalize and convert to tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        return img, labels


def collate_fn_yolo(batch):
    """
    Custom collate function for YOLO-style batches
    
    Args:
        batch: List of (image, labels) tuples
    
    Returns:
        images: Tensor of shape (B, 3, H, W)
        targets: List of label tensors, each of shape (N, 6)
                 where each row is [batch_idx, class_id, x, y, w, h]
    """
    images, labels = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Process labels
    targets = []
    for i, label in enumerate(labels):
        if len(label) > 0:
            # Add batch index as first column
            batch_idx = torch.full((len(label), 1), i, dtype=torch.float32)
            label_tensor = torch.from_numpy(label).float()
            target = torch.cat([batch_idx, label_tensor], dim=1)
            targets.append(target)
    
    if len(targets) > 0:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 6))
    
    return images, targets


# Example usage in your training script
if __name__ == "__main__":
    # Test the dataset
    dataset = NEUDETDatasetYOLO(
        data_dir=r"C:\Users\Dhaksha\ghostcam-bwfpn\NEU-DET-bil",
        split='train',
        input_size=640,
        use_mosaic=True,
        mosaic_prob=0.5,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class names: {dataset.class_names}")
    
    # Set epoch to test mosaic control
    dataset.set_epoch(0)  # Should use mosaic
    img, labels = dataset[0]
    print(f"\nEpoch 0 (with mosaic):")
    print(f"Image shape: {img.shape}")
    print(f"Labels shape: {labels.shape}")
    
    dataset.set_epoch(70)  # Should not use mosaic
    img, labels = dataset[0]
    print(f"\nEpoch 70 (without mosaic):")
    print(f"Image shape: {img.shape}")
    print(f"Labels shape: {labels.shape}")