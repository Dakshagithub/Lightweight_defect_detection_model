# File: utils/logger.py
"""
Training logger with file and console output - WINDOWS COMPATIBLE.
"""

import logging
from pathlib import Path
from datetime import datetime
import json
import sys


class Logger:
    """Custom logger for training - Windows compatible."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        
        # Setup file logger
        log_file = self.log_dir / f'{experiment_name}.log'
        
        # Fix for Windows: Force UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        
        self.logger = logging.getLogger(experiment_name)
        
        # Metrics history
        self.metrics_history = []
    
    def info(self, message: str):
        """Log info message - remove unicode checkmarks for Windows."""
        # Replace unicode checkmarks with ASCII for Windows compatibility
        message = message.replace('✓', '[OK]').replace('★', '[*]').replace('⚠', '[!]')
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        message = message.replace('✓', '[OK]').replace('★', '[*]').replace('⚠', '[!]')
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        message = message.replace('✓', '[OK]').replace('★', '[*]').replace('⚠', '[!]')
        self.logger.error(message)
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """Log epoch metrics."""
        self.info(f"Epoch {epoch} - Train: " + 
                 " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        self.info(f"Epoch {epoch} - Val: " + 
                 " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        self.info(f"Epoch {epoch} - LR: {lr:.6f}")
        
        # Store metrics
        self.metrics_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': lr
        })
    
    def save_metrics(self):
        """Save metrics history to JSON."""
        metrics_file = self.log_dir / 'metrics_history.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.info(f"Metrics saved to {metrics_file}")
    
    def close(self):
        """Close logger and save metrics."""
        self.save_metrics()