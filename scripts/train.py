"""
Training script for PedalNet.
Optimizes the U-Net perception module using a hybrid L1+MSE loss over Mel-spectrograms.
"""

import os
import sys
import yaml
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchaudio.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset import MaestroDataset
from src.models.pedal_net import PedalNet
from src.losses import PedalLoss
from src.utils import set_seed, get_logger, save_checkpoint, count_parameters

def train(cfg):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    exp_name = cfg['experiment']['name']
    log_dir = os.path.join(PROJECT_ROOT, "logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    logger = get_logger(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    logger.info(f"Experiment initialized: {exp_name} on {device}")

    data_root = cfg['data']['root_dir']
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root)
        
    dry_root = cfg['data'].get('dry_dir', None)
    if dry_root and not os.path.isabs(dry_root):
        dry_root = os.path.join(PROJECT_ROOT, dry_root)
        
    if dry_root:
        logger.info(f"Original MIDI Source: {data_root}")
        logger.info(f"Dry Audio Source: {dry_root}")
        if not os.path.exists(dry_root):
            logger.error(f"Dry directory not found: {dry_root}")
            return
    else:
        logger.warning("'dry_dir' not specified. Training will default to legacy wet audio.")

    train_dataset = MaestroDataset(
        root_dir=data_root,
        dry_dir=dry_root, 
        split='train',
        segment_seconds=cfg['data']['segment_seconds'],
        sr=cfg['data']['sr']
    )
    
    val_dataset = MaestroDataset(
        root_dir=data_root,
        dry_dir=dry_root,       
        split='validation',
        segment_seconds=cfg['data']['segment_seconds'],
        sr=cfg['data']['sr']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    model = PedalNet(
        in_channels=cfg['model']['in_channels'],
        out_channels=cfg['model']['out_channels']
    ).to(device)
    
    logger.info(f"Model Parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['num_epochs'])
    criterion = PedalLoss(mse_weight=1.0, l1_weight=0.1).to(device)

    best_val_loss = float('inf')
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=cfg['data']['sr'],
        n_fft=cfg['audio_processing']['n_fft'],
        hop_length=cfg['data']['hop_length'],
        n_mels=cfg['audio_processing']['n_mels'],
        f_min=cfg['audio_processing']['fmin'],
        f_max=cfg['audio_processing']['fmax']
    ).to(device)

    for epoch in range(cfg['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(device)
            target = batch['pedal'].to(device)
            
            with torch.no_grad():
                melspec = mel_transform(audio)
                melspec = torch.log(melspec + 1e-9)
                if melspec.shape[-1] != target.shape[-1]:
                    target = torch.nn.functional.interpolate(target, size=melspec.shape[-1], mode='linear')

            optimizer.zero_grad()
            prediction = model(melspec)
            loss = criterion(prediction, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            global_step = epoch * len(train_loader) + batch_idx
            if global_step % cfg['training']['log_interval'] == 0:
                writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        scheduler.step()
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                target = batch['pedal'].to(device)
                
                melspec = torch.log(mel_transform(audio) + 1e-9)
                if melspec.shape[-1] != target.shape[-1]:
                    target = torch.nn.functional.interpolate(target, size=melspec.shape[-1], mode='linear')
                
                prediction = model(melspec)
                loss = criterion(prediction, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(ckpt_dir, "best_pedal_net.pth")
            save_checkpoint(model, optimizer, epoch, best_val_loss, save_path)
            
        if (epoch + 1) % cfg['training']['save_interval'] == 0:
            save_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, avg_val_loss, save_path)

    writer.close()
    logger.info("Training Execution Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PedalNet")
    default_config = os.path.join(PROJECT_ROOT, 'configs', 'default.yaml')
    parser.add_argument('--config', type=str, default=default_config, help='Path to configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    train(config)