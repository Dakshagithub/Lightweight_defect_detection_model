# File: train.py
"""
Main Training Script with All Optimizations (Phase 1, 2, 3, 4).

Includes:
- Phase 1: C-NMS + Post-processing
- Phase 2: Higher objectness weight + Focal Loss
- Phase 3: Optimized anchors
- Phase 4: EIoU loss + Warmup scheduler
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models import GhostCAMBWFPN
# from datasets import NEUDETDataset, collate_fn, get_train_transforms, get_val_transforms
from datasets.neu_det_yolo import NEUDETDatasetYOLO, collate_fn_yolo
from utils import (
    YOLOLoss,
    Logger,
    save_checkpoint,
    load_checkpoint,
    ModelEMA,
    plot_training_curves
)
# Import optimize_anchors function directly to avoid naming conflict
from utils.optimize_anchors import optimize_anchors as run_anchor_optimization


class Trainer:
    """
    Trainer class with all optimizations.
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Setup directories
        self.setup_directories()
        
        # Initialize logger
        self.logger = Logger(
            log_dir=self.exp_dir / 'logs',
            experiment_name=config['experiment_name']
        )
        
        # Log configuration
        self.logger.info("=" * 60)
        self.logger.info("Training Configuration")
        self.logger.info("=" * 60)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
        
        # Training state - MOVED HERE EARLY
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.train_maps = []
        self.val_maps = []
        self.learning_rates = []
        
        # PHASE 3: Optimize anchors
        self.optimized_anchors = self.optimize_anchors()
        
        # Build model with optimized anchors
        self.build_model()
        
        # Build datasets
        self.build_datasets()
        
        # Build optimizer and scheduler
        self.build_optimizer()
        
        # Build loss function
        self.criterion = YOLOLoss(
            num_classes=config['num_classes'],
            img_size=config['input_size'],
            lambda_box=config['lambda_box'],
            lambda_obj=config['lambda_obj'],
            lambda_cls=config['lambda_cls'],
            label_smoothing=config['label_smoothing']
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'])
    
    def setup_directories(self):
        """Setup experiment directories."""
        self.exp_dir = Path('experiments') / self.config['experiment_name']
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.results_dir = Path('results')
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = self.exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def optimize_anchors(self):
        """PHASE 3: Optimize anchors for NEU-DET dataset."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 3: Anchor Optimization")
        self.logger.info("=" * 60)
        
        optimized_anchors = run_anchor_optimization(
            data_dir=self.config['data_dir'],  # Use the correct data directory
            num_anchors=9,
            input_size=self.config['input_size']
        )
        
        self.logger.info("Optimized anchors will be used for training")
        
        return optimized_anchors
    
    def build_model(self):
        """Build model with optimized anchors."""
        self.logger.info("\nBuilding model...")
        
        self.model = GhostCAMBWFPN(
            num_classes=self.config['num_classes'],
            in_channels=3,
            anchors=self.optimized_anchors  # Use optimized anchors
        )
        self.model.to(self.device)
        
        # Model EMA
        if self.config['ema_decay'] > 0:
            self.ema = ModelEMA(
                self.model,
                decay=self.config['ema_decay'],
                device=self.device
            )
            self.logger.info(f"✓ EMA enabled with decay: {self.config['ema_decay']}")
        else:
            self.ema = None
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"✓ Total parameters: {total_params / 1e6:.2f}M")
        self.logger.info(f"✓ Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Load checkpoint if specified
        if self.config.get('resume_from'):
            checkpoint_info = load_checkpoint(
                checkpoint_path=self.config['resume_from'],
                model=self.model,
                optimizer=None,
                ema=self.ema,
                device=self.device
            )
            self.start_epoch = checkpoint_info['epoch']
            self.logger.info(f"✓ Resumed from epoch {self.start_epoch}")
    
    def build_datasets(self):
        """Build train and validation datasets."""
        self.logger.info("\nBuilding datasets...")
        
        # Training dataset

        train_dataset = NEUDETDatasetYOLO(
            data_dir=self.config['data_dir'],
            split='train',
            input_size=self.config['input_size'],
            use_mosaic=self.config['augmentation'].get('mosaic_prob', 0.0) > 0,
            mosaic_prob=self.config['augmentation'].get('mosaic_prob', 0.7),
            augment=True
        )
        
        # Validation dataset
      
        val_dataset = NEUDETDatasetYOLO(
            data_dir=self.config['data_dir'],
            split='validation',
            input_size=self.config['input_size'],
            use_mosaic=False, # No mosaic for validation
            mosaic_prob=0.0,
            augment=False
        )
        
        self.logger.info(f"✓ Train dataset: {len(train_dataset)} images")
        self.logger.info(f"✓ Val dataset: {len(val_dataset)} images")
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=collate_fn_yolo,
            pin_memory=self.config['pin_memory'],
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=collate_fn_yolo,
            pin_memory=self.config['pin_memory']
        )
        
        self.class_names = train_dataset.class_names
        self.train_dataset = train_dataset # Store for epoch setting
    
    def build_optimizer(self):
        """Build optimizer and learning rate scheduler with warmup (Phase 4.3)."""
        self.logger.info("\nBuilding optimizer and scheduler...")
        
        # Optimizer
        if self.config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Load optimizer state if resuming
        if self.config.get('resume_from'):
            checkpoint = torch.load(self.config['resume_from'], map_location=self.device)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✓ Optimizer state loaded")
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['T_0'],
                T_mult=self.config['T_mult'],
                eta_min=self.config['eta_min']
            )
        elif self.config['scheduler'] == 'CosineAnnealingLR':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['eta_min']
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config['scheduler']}")
        
        # PHASE 4.3: Add warmup scheduler
        if self.config.get('warmup_epochs', 0) > 0:
            warmup_epochs = self.config['warmup_epochs']
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.get('warmup_start_factor', 0.1),
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            
            self.logger.info(f"✓ Warmup scheduler added for {warmup_epochs} epochs")
        else:
            self.scheduler = main_scheduler
        
        # Adjust scheduler if resuming
        if self.start_epoch > 0:
            for _ in range(self.start_epoch):
                self.scheduler.step()
        
        self.logger.info(f"✓ Optimizer: {self.config['optimizer']}")
        self.logger.info(f"✓ Scheduler: {self.config['scheduler']}")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)
            
            # Accumulate losses
            total_loss += loss_dict['loss']
            total_box_loss += loss_dict['loss_box']
            total_obj_loss += loss_dict['loss_obj']
            total_cls_loss += loss_dict['loss_cls']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'box': f"{loss_dict['loss_box']:.4f}",
                'obj': f"{loss_dict['loss_obj']:.4f}",
                'cls': f"{loss_dict['loss_cls']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate average losses
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate the model."""
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        
        total_loss = 0.0
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Validation")
        
        for images, targets in pbar:
            images = images.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
                predictions = model(images)
                loss, loss_dict = self.criterion(predictions, targets, self.model)
            
            total_loss += loss_dict['loss']
            total_box_loss += loss_dict['loss_box']
            total_obj_loss += loss_dict['loss_obj']
            total_cls_loss += loss_dict['loss_cls']
            
            pbar.set_postfix({'loss': f"{loss_dict['loss']:.4f}"})
        
        num_batches = len(self.val_loader)
        
        # Return metrics (mAP disabled during training for speed)
        metrics = {
            'loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'mAP@0.5': 0.0,
            'mAP@0.75': 0.0,
            'mAP@[0.5:0.95]': 0.0
        }
        
        return metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        self.logger.info("Using validation LOSS for early stopping")
        self.logger.info("Run evaluate.py after training for full metrics")
        self.logger.info("=" * 60)
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Set epoch for mosaic augmentation control in YOLO dataset
            if hasattr(self.train_dataset, 'set_epoch'):
                self.train_dataset.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_metrics = self.validate(epoch + 1)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.log_epoch(
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr
            )
            
            # Store for plotting
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_maps.append(0.0)
            self.val_maps.append(0.0)
            self.learning_rates.append(current_lr)
            
            # Check if best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.logger.info(f"★ New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                metrics=val_metrics,
                checkpoint_dir=self.checkpoint_dir,
                filename=f'epoch_{epoch + 1}.pth',
                ema=self.ema,
                is_best=is_best
            )
            
            # Update learning rate
            self.scheduler.step()
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                self.logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"No improvement for {self.config['early_stopping_patience']} epochs")
                self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                break
            
            # Plot training curves
            if (epoch + 1) % 10 == 0 or is_best:
                plot_training_curves(
                    self.train_losses,
                    self.val_losses,
                    self.train_maps,
                    self.val_maps,
                    self.learning_rates,
                    str(self.results_dir / 'training_curves.png')
                )
        
        # Final logging
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training Completed")
        self.logger.info("=" * 60)
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best model: {self.checkpoint_dir / 'best.pth'}")
        self.logger.info("\nTo evaluate metrics, run:")
        self.logger.info(f"  python evaluate.py --checkpoint {self.checkpoint_dir / 'best.pth'}")
        self.logger.info("=" * 60)
        
        # Close logger
        self.logger.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GhostCAM-BWFPN')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.resume_from:
        config['resume_from'] = args.resume_from
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Print device info
    if args.device == 'cuda':
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create trainer and start training
    trainer = Trainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()