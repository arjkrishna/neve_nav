#!/usr/bin/env python3
"""
create_training_collage.py

Creates a 3x2 grid collage of training metric plots from a given experiment.

Usage:
    python create_training_collage.py --name diag_debug_test9
    python create_training_collage.py --name diag_debug_test9 --output training_collage.png
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_experiment_dir(name: str, results_base: str = "training_scripts/results/eve_paper/neurovascular/full/mesh_ben") -> Path:
    """Find experiment directory by name."""
    base_dir = Path(results_base).expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Results base dir does not exist: {base_dir}")
    
    matching = [d for d in base_dir.iterdir() if d.is_dir() and name in d.name]
    if not matching:
        raise SystemExit(f"No folder found matching name '{name}'")
    
    return max(matching, key=lambda p: p.stat().st_mtime)

def create_training_collage(experiment_dir: Path, output_path: Path = None):
    """Create a 3x2 grid collage of training metric plots."""
    analysis_dir = experiment_dir / "diagnostics" / "analysis"
    
    if not analysis_dir.exists():
        raise SystemExit(f"Analysis directory does not exist: {analysis_dir}")
    
    # Define the 5 training plots in desired order (6th slot will be empty)
    # Layout: Left column (pref, reward, trans), Right column (probe mean, probe std, empty)
    plot_files = [
        "batch_q_preference_zoomed_v4.png",  # Top left: Critic preference (zoomed)
        "probe_trans_mean_v4.png",           # Top right: Probe trans mean
        "batch_reward_v4.png",               # Middle left: Batch reward mean
        "probe_trans_std_v4.png",            # Middle right: Probe trans std
        "batch_trans_v4.png",                # Bottom left: Batch trans mean
        None                                 # Bottom right: Empty slot
    ]
    
    # Check which files exist
    existing_plots = []
    missing_plots = []
    
    for plot_file in plot_files:
        if plot_file is None:
            # Skip None entries (empty slots)
            continue
        plot_path = analysis_dir / plot_file
        if plot_path.exists():
            existing_plots.append((plot_file, plot_path))
        else:
            missing_plots.append(plot_file)
    
    if not existing_plots:
        raise SystemExit(f"No training plots found in {analysis_dir}")
    
    if missing_plots:
        print(f"Warning: Missing plots: {missing_plots}")
    
    # Create figure with 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(f"Training Metrics Collage\n{experiment_dir.name}", fontsize=16, fontweight='bold')
    
    # Load and display each plot
    plot_idx = 0
    for idx in range(6):  # 6 positions total
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if idx < len(existing_plots):
            # Load and display plot
            plot_file, plot_path = existing_plots[plot_idx]
            img = mpimg.imread(str(plot_path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(plot_file.replace('_v4.png', '').replace('_', ' ').title(), 
                        fontsize=12, fontweight='bold', pad=10)
            plot_idx += 1
        else:
            # Empty slot - just hide the axis
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
    
    # Save collage
    if output_path is None:
        output_path = analysis_dir / "training_collage.png"
    else:
        output_path = Path(output_path).expanduser().resolve()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Collage saved to: {output_path}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Create 3x2 grid collage of training metric plots")
    ap.add_argument("--name", type=str, required=True, help="Experiment name (e.g., diag_debug_test9)")
    ap.add_argument("--results-base", type=str, 
                   default="training_scripts/results/eve_paper/neurovascular/full/mesh_ben",
                   help="Base results directory")
    ap.add_argument("--output", type=str, help="Output path for collage (default: diagnostics/analysis/training_collage.png)")
    args = ap.parse_args()
    
    # Find experiment directory
    experiment_dir = find_experiment_dir(args.name, args.results_base)
    print(f"Found experiment: {experiment_dir}")
    
    # Create collage
    create_training_collage(experiment_dir, args.output)

if __name__ == "__main__":
    main()
