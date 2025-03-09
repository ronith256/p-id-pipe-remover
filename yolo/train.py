# train.py - YOLOv11 P&ID Symbol Detector Training
# 
# This script trains a YOLOv11 model on the P&ID symbols dataset
# using the local files and CUDA acceleration

import os
import torch
from ultralytics import YOLO

def main():
    print("Starting YOLOv11 training for P&ID symbol detection")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: CUDA not available. Training will be slow on CPU.")
    
    # Create results directory
    os.makedirs('runs/train', exist_ok=True)
    
    # Initialize YOLOv11 model
    model = YOLO('yolo11n.pt')  # Using nano model for faster training
    
    # Train the model
    results = model.train(
        data='dataset.yaml',          # Path to dataset YAML
        epochs=35,                    # Number of epochs
        imgsz=672,                    # Image size
        batch=8,                      # Batch size
        device=0 if device.type == 'cuda' else 'cpu',  # Use GPU 0 if available
        workers=8,                    # Number of worker threads
        patience=5,                   # Early stopping patience
        save=True,                    # Save checkpoints
        project='pid_symbol_detector', # Project name
        name='yolov11_training',      # Run name
        exist_ok=True,                # Overwrite existing run
        pretrained=True,              # Use pretrained weights
        optimizer='auto',             # Optimizer (SGD, Adam, etc.)
        verbose=True,                 # Print verbose output
        seed=42,                      # Random seed for reproducibility
        deterministic=True,           # Use deterministic algorithms
        rect=False,                   # Use rectangular training
        cos_lr=True,                  # Use cosine learning rate scheduler
        close_mosaic=10,              # Close mosaic augmentation in last 10 epochs
        resume=False,                 # Resume training from last checkpoint
        amp=True,                     # Use mixed precision training
        lr0=0.01,                     # Initial learning rate
        lrf=0.01,                     # Final learning rate factor
        warmup_epochs=3.0,            # Warmup epochs
        warmup_momentum=0.8,          # Warmup momentum
        warmup_bias_lr=0.1,           # Warmup bias learning rate
        multi_scale=True,             # Multi-scale training
        label_smoothing=0.0,          # Label smoothing epsilon
        nbs=64,                       # Nominal batch size
        overlap_mask=True,            # Masks should overlap during training
        weight_decay=0.0005,          # Weight decay
        dropout=0.0,                  # Use dropout
        val=True,                     # Validate during training
    )
    
    # Validate the trained model
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
    
    # Print training summary
    print("\nTraining completed!")
    print(f"Best model saved at: {model.best}")
    
    return model

if __name__ == "__main__":
    main()