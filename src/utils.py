"""
Utility functions for experiment tracking, reproducibility, and checkpoint management.
"""

import os
import torch
import numpy as np
import random
import logging
import time

def set_seed(seed=42):
    """Ensures deterministic behavior for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed globally set to {seed}")

def get_logger(log_dir, name="experiment"):
    """Creates a standardized logger for both console and file output."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
        
    return logger

def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, best_loss, path):
    """Serializes the model state to disk."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, path)
    logging.info(f"Checkpoint successfully saved to {path}")

def load_checkpoint(path, model, optimizer=None, device='cuda'):
    """Deserializes the model state from disk."""
    if not os.path.exists(path):
        logging.warning(f"Checkpoint file not found at {path}. Starting from scratch.")
        return None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Loaded checkpoint from {path} (Epoch {checkpoint.get('epoch', 0)})")
    return checkpoint