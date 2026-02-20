"""Main training script for chess CNN model."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
from tqdm import tqdm

from config import *
from model import ChessCNN, ChessCNNv2
from dataset import get_dataloaders
from utils import AverageMeter, ProgressMeter, save_checkpoint, accuracy_topk, log_metrics


def train_epoch(train_loader, model, criterion, optimizer, epoch, device, log_interval=100):
    """Train for one epoch."""
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':.4f')
    top5 = AverageMeter('Acc@5', ':.4f')

    model.train()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            # Forward pass
            output = model(images)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            acc1, acc5 = accuracy_topk(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc@1': f'{top1.avg:.2f}',
                'acc@5': f'{top5.avg:.2f}'
            })

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device):
    """Validate the model."""
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':.4f')
    top5 = AverageMeter('Acc@5', ':.4f')

    model.eval()

    with tqdm(total=len(val_loader), desc='Validating', unit='batch') as pbar:
        with torch.no_grad():
            for images, target in val_loader:
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                acc1, acc5 = accuracy_topk(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'acc@1': f'{top1.avg:.2f}',
                    'acc@5': f'{top5.avg:.2f}'
                })

    return losses.avg, top1.avg, top5.avg


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='Train chess CNN model')
    parser.add_argument('--model', choices=['v1', 'v2'], default='v1', help='Model version')
    parser.add_argument('--dataset-dir', default=DATASET_DIR, help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--device', default=None, help='Device (cuda, mps, cpu). Auto-detect if not specified')
    parser.add_argument('--checkpoint', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Create experiment log directory with timestamp and key parameters
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"exp_{timestamp}_model-{args.model}_bs-{args.batch_size}_lr-{args.lr}_epochs-{args.epochs}"
    log_dir = Path("logs") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging experiment to: {log_dir}")

    # Setup device with auto-detection for Apple Silicon
    if args.device:
        device = torch.device(args.device)
    else:
        # Auto-detect best device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Print device info
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    elif device.type == 'mps':
        print("GPU: Apple Metal Performance Shaders (MPS)")

    # Create model
    if args.model == 'v1':
        model = ChessCNN()
    else:
        model = ChessCNNv2()
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    print(f"Loading data from {args.dataset_dir}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        max_games_per_file=GAMES_PER_FILE
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # Train
        train_loss, train_acc, train_acc5 = train_epoch(
            train_loader, model, criterion, optimizer, epoch, device,
            log_interval=LOG_INTERVAL
        )

        # Validate
        print('Validation: ', end='')
        val_loss, val_acc, val_acc5 = validate(val_loader, model, criterion, device)

        # Log metrics to experiment folder
        log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_acc5, val_acc5, log_dir=log_dir)

        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, CHECKPOINT_DIR)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, CHECKPOINT_DIR, 
                          name='best_model.pt')

        scheduler.step()

    # Test on test set
    print('\nTesting: ', end='')
    test_loss, test_acc, test_acc5 = validate(test_loader, model, criterion, device)
    print(f"Test Accuracy@1: {test_acc:.4f}, Accuracy@5: {test_acc5:.4f}")


if __name__ == '__main__':
    main()
