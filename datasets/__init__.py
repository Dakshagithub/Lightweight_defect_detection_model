# File: datasets/__init__.py
"""
Datasets package for NEU-DET steel surface defect detection.
"""

# The old 'neudet_dataset.py' is missing or incorrect.
# We will import the correct YOLO-based dataset and collate function.
from .neu_det_yolo import NEUDETDatasetYOLO, collate_fn_yolo
from .augmentations import get_train_transforms, get_val_transforms

# Rename for backward compatibility with train.py and evaluate.py
NEUDETDataset = NEUDETDatasetYOLO
collate_fn = collate_fn_yolo

__all__ = [
    'NEUDETDataset', 'collate_fn', 'get_train_transforms', 'get_val_transforms'
]