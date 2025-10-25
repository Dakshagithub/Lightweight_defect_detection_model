# File: utils/optimize_anchors.py
"""
Optimize anchor boxes for NEU-DET dataset using K-means clustering.
This is Phase 3 of the optimization strategy.
"""

import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def extract_box_sizes(data_dir: str, split: str = 'train') -> np.ndarray:
    """Extract all box widths and heights from dataset."""
    # Updated to read YOLO format .txt files from the 'labels' directory
    labels_dir = Path(data_dir) / split / 'annotations' / 'labels'
    
    sizes = []
    for label_file in labels_dir.glob('*.txt'):
        try:
            # Each line is: class_id, x_center, y_center, width, height
            labels = np.loadtxt(label_file, ndmin=2)
            
            if labels.shape[1] == 5:
                # Extract width and height (columns 3 and 4)
                # These are normalized, so we don't need to scale them yet.
                # K-means will work on the normalized dimensions.
                w_h = labels[:, 3:5]
                sizes.extend(w_h.tolist())

        except Exception as e:
            print(f"Error parsing {label_file}: {e}")
            continue
    
    if not sizes:
        print(f"Warning: No valid bounding boxes found in {labels_dir}")
        return np.array([])

    return np.array(sizes)


def optimize_anchors(
    data_dir: str,
    num_anchors: int = 9,
    input_size: int = 200
) -> list:
    """
    Use K-means clustering to find optimal anchor sizes.
    
    Returns optimized anchors grouped into 3 scales (P3, P4, P5).
    """
    print("=" * 70)
    print("PHASE 3: Optimizing Anchors for NEU-DET Dataset")
    print("=" * 70)
    
    # Extract box sizes from training data
    sizes = extract_box_sizes(data_dir, split='train')
    
    if sizes.shape[0] == 0:
        print("Error: No boxes found. Cannot optimize anchors. Exiting.")
        # Return default COCO anchors as a fallback
        return [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]
        ]

    print(f"\n✓ Extracted {len(sizes)} boxes from training set")
    # The sizes are normalized, so we scale them by input_size for display
    pixel_sizes = sizes * input_size
    print(f"  Width range (pixels):  [{pixel_sizes[:, 0].min():.1f}, {pixel_sizes[:, 0].max():.1f}]")
    print(f"  Height range (pixels): [{pixel_sizes[:, 1].min():.1f}, {pixel_sizes[:, 1].max():.1f}]")
    
    # K-means clustering
    print(f"\n✓ Running K-means with {num_anchors} clusters...")
    kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=20)
    kmeans.fit(sizes)
    
    anchors = kmeans.cluster_centers_

    # Scale anchors to pixel values
    anchors *= input_size
    
    # Sort by area (small to large)
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])
    
    # Group into 3 scales
    anchors_p3 = [tuple(map(int, a)) for a in anchors[0:3]]
    anchors_p4 = [tuple(map(int, a)) for a in anchors[3:6]]
    anchors_p5 = [tuple(map(int, a)) for a in anchors[6:9]]
    
    print("\n" + "=" * 70)
    print("✓ OPTIMIZED ANCHORS (width, height):")
    print("=" * 70)
    print(f"P3 (small objects):  {anchors_p3}")
    print(f"P4 (medium objects): {anchors_p4}")
    print(f"P5 (large objects):  {anchors_p5}")
    
    # Compare with default COCO anchors
    coco_anchors = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)]
    ]
    
    print("\n" + "=" * 70)
    print("Default COCO Anchors (for comparison):")
    print("=" * 70)
    print(f"P3: {coco_anchors[0]}")
    print(f"P4: {coco_anchors[1]}")
    print(f"P5: {coco_anchors[2]}")
    
    # Calculate improvement metrics
    print("\n" + "=" * 70)
    print("Anchor Quality Metrics:")
    print("=" * 70)
    
    # Average IoU with ground truth
    optimized_flat = np.array(anchors)
    coco_flat = np.array(coco_anchors[0] + coco_anchors[1] + coco_anchors[2])
    
    def compute_avg_iou(boxes, anchors):
        """Compute average best IoU between boxes and anchors. Assumes pixel values."""
        ious = []
        for box in boxes:
            box_area = box[0] * box[1]
            best_iou = 0
            for anchor in anchors:
                anchor_area = anchor[0] * anchor[1]
                inter_w = min(box[0], anchor[0])
                inter_h = min(box[1], anchor[1])
                inter_area = inter_w * inter_h
                union_area = box_area + anchor_area - inter_area
                iou = inter_area / (union_area + 1e-6)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)
        return np.mean(ious)
    
    opt_iou = compute_avg_iou(pixel_sizes, optimized_flat)
    coco_iou = compute_avg_iou(pixel_sizes, coco_flat)
    
    print(f"  Optimized anchors - Avg IoU: {opt_iou:.4f}")
    print(f"  COCO anchors      - Avg IoU: {coco_iou:.4f}")
    print(f"  Improvement:               {((opt_iou - coco_iou) / coco_iou * 100):+.2f}%")
    
    # Visualize
    visualize_anchors(pixel_sizes, optimized_flat, coco_flat)
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. These optimized anchors will be automatically used in training")
    print("2. Check 'anchor_optimization.png' for visualization")
    print("3. Start training with: python train.py")
    print("=" * 70)
    
    return [anchors_p3, anchors_p4, anchors_p5]


def visualize_anchors(sizes: np.ndarray, optimized: np.ndarray, coco: np.ndarray):
    """Visualize box distribution and anchor placement."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Box size distribution with anchors
    axes[0].scatter(sizes[:, 0], sizes[:, 1], alpha=0.3, s=10, c='gray', label='Ground Truth Boxes')
    axes[0].scatter(optimized[:, 0], optimized[:, 1], 
                   c='red', s=300, marker='X', edgecolors='black', linewidths=2.5,
                   label='Optimized Anchors', zorder=5)
    axes[0].scatter(coco[:, 0], coco[:, 1], 
                   c='blue', s=300, marker='s', edgecolors='black', linewidths=2.5,
                   label='COCO Anchors', zorder=5, alpha=0.7)
    axes[0].set_xlabel('Width (pixels)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Height (pixels)', fontsize=12, fontweight='bold')
    axes[0].set_title('Box Size Distribution & Anchors', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 210)
    axes[0].set_ylim(0, 210)
    
    # Plot 2: Aspect ratio distribution
    ratios = sizes[:, 0] / (sizes[:, 1] + 1e-6)
    axes[1].hist(ratios, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].axvline(1.0, color='red', linestyle='--', linewidth=2.5, label='Square (1:1)')
    axes[1].set_xlabel('Aspect Ratio (W/H)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend(fontsize=10)
    
    # Plot 3: Area distribution
    areas = sizes[:, 0] * sizes[:, 1]
    axes[2].hist(areas, bins=50, alpha=0.7, edgecolor='black', color='coral')
    axes[2].set_xlabel('Box Area (pixels²)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[2].set_title('Box Area Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('anchor_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: anchor_optimization.png")
    plt.close()


if __name__ == '__main__':
    # Optimize anchors for NEU-DET
    optimized_anchors = optimize_anchors(
        data_dir='/home/daksha/projects/ghostcam-bwfpn/NEU-DET',
        num_anchors=9,
        input_size=200
    )