import torch
import os
import glob
import re
from pathlib import Path
from typing import Union

def load_checkpoint(checkpoint_path: Union[str, Path], device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_file():
        return torch.load(checkpoint_path, map_location=device), checkpoint_path
    
    elif checkpoint_path.is_dir():
        checkpoint_files = list(checkpoint_path.glob("*.pt"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_path}")

        checkpoint_info = []
        for file_path in checkpoint_files:
            filename = file_path.stem
            match = re.match(r'^(\d+)_([0-9.]+)$', filename)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                checkpoint_info.append((file_path, epoch, loss))
        
        if not checkpoint_info:
            raise ValueError(f"No valid checkpoint files found in directory: {checkpoint_path}")
        
        checkpoint_info.sort(key=lambda x: x[2])
        best_checkpoint = checkpoint_info[0]
        best_file, best_epoch, best_loss = best_checkpoint
        
        return torch.load(best_file, map_location=device, weights_only=False), best_file
    
    else:
        raise FileNotFoundError(f"Path does not exist: {checkpoint_path}")

def save_checkpoint(checkpoint, checkpoint_path, keep_best_k=5):
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.suffix == '.pt':
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path    
    else:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        
        filename = f"{epoch}_{loss:.4f}.pt"
        checkpoint_save_path = checkpoint_path / filename

        torch.save(checkpoint, checkpoint_save_path)
        
        checkpoint_files = list(checkpoint_path.glob("*.pt"))
        
        if len(checkpoint_files) > keep_best_k:
            checkpoint_info = []
            for file_path in checkpoint_files:
                filename = file_path.stem
                match = re.match(r'^(\d+)_([0-9.]+)$', filename)
                if match:
                    file_epoch = int(match.group(1))
                    file_loss = float(match.group(2))
                    checkpoint_info.append((file_path, file_epoch, file_loss))
            
            checkpoint_info.sort(key=lambda x: x[2])
            files_to_keep = checkpoint_info[:keep_best_k]
            files_to_remove = checkpoint_info[keep_best_k:]
            
            for file_path, _, _ in files_to_remove:
                file_path.unlink()
        
        return checkpoint_save_path
        