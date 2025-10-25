# File: datasets/augmentations.py
"""
Data augmentation pipelines for training and validation.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(input_size: int = 200) -> A.Compose:
    """
    Training augmentation pipeline with strong augmentations.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.9),
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.5),
        # Geometric transforms
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=1.0
    ))


def get_val_transforms(input_size: int = 200) -> A.Compose:
    """
    Validation transforms - only resize and normalize.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=1.0
    ))