"""Generate plots from training logs and test results."""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_metrics(log_file="./logs/metrics.jsonl"):
    """Load training metrics from JSONL file."""
    metrics = []
    
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return None
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None
    
    return pd.DataFrame(metrics) if metrics else None


def plot_loss_curves(df, output_dir="./logs"):
    """Plot training and validation loss."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "loss_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_curves(df, output_dir="./logs"):
    """Plot training and validation accuracy (top-1 and top-5)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['epoch'], df['train_acc'], label='Train Acc@1', marker='o', linewidth=2)
    ax.plot(df['epoch'], df['val_acc'], label='Val Acc@1', marker='s', linewidth=2)
    
    # Plot top-5 if available
    if 'train_acc5' in df.columns:
        ax.plot(df['epoch'], df['train_acc5'], label='Train Acc@5', marker='^', linewidth=2, linestyle='--')
    if 'val_acc5' in df.columns:
        ax.plot(df['epoch'], df['val_acc5'], label='Val Acc@5', marker='d', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = Path(output_dir) / "accuracy_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined(df, output_dir="./logs"):
    """Plot loss and accuracy side-by-side."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(df['epoch'], df['train_acc'], label='Train Acc@1', marker='o', linewidth=2)
    ax2.plot(df['epoch'], df['val_acc'], label='Val Acc@1', marker='s', linewidth=2)
    
    # Plot top-5 if available
    if 'train_acc5' in df.columns:
        ax2.plot(df['epoch'], df['train_acc5'], label='Train Acc@5', marker='^', linewidth=2, linestyle='--')
    if 'val_acc5' in df.columns:
        ax2.plot(df['epoch'], df['val_acc5'], label='Val Acc@5', marker='d', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = Path(output_dir) / "training_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(df):
    """Print training summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\nTotal Epochs: {len(df)}")
    print(f"\nFinal Results:")
    print(f"  Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Val Loss:   {df['val_loss'].iloc[-1]:.4f}")
    print(f"  Train Acc@1: {df['train_acc'].iloc[-1]:.2f}%")
    print(f"  Val Acc@1:   {df['val_acc'].iloc[-1]:.2f}%")
    
    if 'train_acc5' in df.columns:
        print(f"  Train Acc@5: {df['train_acc5'].iloc[-1]:.2f}%")
        print(f"  Val Acc@5:   {df['val_acc5'].iloc[-1]:.2f}%")
    
    print(f"\nBest Results:")
    best_train_loss_idx = df['train_loss'].idxmin()
    best_val_loss_idx = df['val_loss'].idxmin()
    best_train_acc_idx = df['train_acc'].idxmax()
    best_val_acc_idx = df['val_acc'].idxmax()
    
    print(f"  Best Train Loss: {df['train_loss'].min():.4f} (Epoch {best_train_loss_idx})")
    print(f"  Best Val Loss:   {df['val_loss'].min():.4f} (Epoch {best_val_loss_idx})")
    print(f"  Best Train Acc@1: {df['train_acc'].max():.2f}% (Epoch {best_train_acc_idx})")
    print(f"  Best Val Acc@1:   {df['val_acc'].max():.2f}% (Epoch {best_val_acc_idx})")
    
    if 'train_acc5' in df.columns:
        best_train_acc5_idx = df['train_acc5'].idxmax()
        best_val_acc5_idx = df['val_acc5'].idxmax()
        print(f"  Best Train Acc@5: {df['train_acc5'].max():.2f}% (Epoch {best_train_acc5_idx})")
        print(f"  Best Val Acc@5:   {df['val_acc5'].max():.2f}% (Epoch {best_val_acc5_idx})")
    
    print("\n" + "="*60)


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description='Generate training plots')
    parser.add_argument('--exp-dir', default=None,
                       help='Experiment log directory (default: latest under logs/)')
    parser.add_argument('--plots', choices=['all', 'loss', 'accuracy', 'combined'], 
                       default='all', help='Which plots to generate')
    args = parser.parse_args()

    # Determine experiment directory
    logs_root = Path('logs')
    if args.exp_dir is None:
        # Find latest experiment folder
        exp_dirs = [d for d in logs_root.iterdir() if d.is_dir() and d.name.startswith('exp_')]
        if not exp_dirs:
            print('No experiment folders found in logs/.')
            return
        exp_dir = max(exp_dirs, key=lambda d: d.stat().st_mtime)
        print(f"Using latest experiment folder: {exp_dir}")
    else:
        exp_dir = Path(args.exp_dir)
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            return

    log_file = exp_dir / 'metrics.jsonl'
    output_dir = exp_dir

    # Load metrics
    print(f"Loading metrics from {log_file}...")
    df = load_metrics(log_file)

    if df is None or df.empty:
        print("No metrics to plot.")
        return

    # Print summary
    print_summary(df)

    # Generate plots
    print(f"\nGenerating plots in {output_dir}...")

    if args.plots in ['all', 'loss']:
        plot_loss_curves(df, output_dir)

    if args.plots in ['all', 'accuracy']:
        plot_accuracy_curves(df, output_dir)

    if args.plots in ['all', 'combined']:
        plot_combined(df, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
