#!/usr/bin/env python3
"""
create_episode_collage.py

Creates a 3x2 grid collage of episode plots from a given experiment.

Usage:
    python create_episode_collage.py --name diag_debug_test9
    python create_episode_collage.py --name diag_debug_test9 --output episode_collage.png
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

def create_episode_collage(experiment_dir: Path, output_path: Path = None):
    """Create a 3x2 grid collage of episode plots."""
    analysis_dir = experiment_dir / "diagnostics" / "analysis"
    
    if not analysis_dir.exists():
        raise SystemExit(f"Analysis directory does not exist: {analysis_dir}")
    
    # Define the 6 episode plots in desired order
    plot_files = [
        "insertion_v4.png",
        "episode_rewards_v4.png",
        "translation_speed_v4.png",
        "stuck_instances_v4.png",
        "stuck_fraction_v4.png",
        "stuck_max_run_v4.png"
    ]
    
    # Check which files exist
    existing_plots = []
    missing_plots = []
    
    for plot_file in plot_files:
        plot_path = analysis_dir / plot_file
        if plot_path.exists():
            existing_plots.append((plot_file, plot_path))
        else:
            missing_plots.append(plot_file)
    
    if not existing_plots:
        raise SystemExit(f"No episode plots found in {analysis_dir}")
    
    if missing_plots:
        print(f"Warning: Missing plots: {missing_plots}")
    
    # Limit to 6 plots (or fewer if some are missing)
    existing_plots = existing_plots[:6]
    
    # Create figure with 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(f"Episode Analysis Collage\n{experiment_dir.name}", fontsize=16, fontweight='bold')
    
    # Load and display each plot
    for idx, (plot_file, plot_path) in enumerate(existing_plots):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Load image
        img = mpimg.imread(str(plot_path))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(plot_file.replace('_v4.png', '').replace('_', ' ').title(), 
                    fontsize=12, fontweight='bold', pad=10)
    
    # Hide unused subplots if we have fewer than 6 plots
    for idx in range(len(existing_plots), 6):
        row = idx // 2
        col = idx % 2
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
    
    # Save collage
    if output_path is None:
        output_path = analysis_dir / "episode_collage.png"
    else:
        output_path = Path(output_path).expanduser().resolve()
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Collage saved to: {output_path}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Create 3x2 grid collage of episode plots")
    ap.add_argument("--name", type=str, required=True, help="Experiment name (e.g., diag_debug_test9)")
    ap.add_argument("--results-base", type=str, 
                   default="training_scripts/results/eve_paper/neurovascular/full/mesh_ben",
                   help="Base results directory")
    ap.add_argument("--output", type=str, help="Output path for collage (default: diagnostics/analysis/episode_collage.png)")
    args = ap.parse_args()
    
    # Find experiment directory
    experiment_dir = find_experiment_dir(args.name, args.results_base)
    print(f"Found experiment: {experiment_dir}")
    
    # Create collage
    create_episode_collage(experiment_dir, args.output)

if __name__ == "__main__":
    main()
