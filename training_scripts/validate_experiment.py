#!/usr/bin/env python3
"""
validate_experiment.py

Validates all log files, diagnostics, and probes for a given experiment,
checking completeness and data integrity.

Supports both completed and in-progress experiments:
- Detects current training phase (heatup, training, evaluation)
- Checks file timestamps for activity
- Provides context for missing files during early phases
- Optionally checks Docker container logs

Usage:
    python validate_experiment.py diag_debug_test5
    python validate_experiment.py diag_debug_test5 --verbose
    python validate_experiment.py --full-path /path/to/experiment
    python validate_experiment.py diag_debug_test5 --docker-container my_container
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Try to import numpy for probe_states validation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import yaml for config validation
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Training Phase Detection
# =============================================================================

@dataclass
class TrainingStatus:
    """Current status of the training run."""
    phase: str  # "unknown", "heatup", "training", "evaluation", "completed", "crashed"
    is_active: bool
    last_activity: Optional[datetime] = None
    activity_age_minutes: float = 0.0
    heatup_steps: int = 0
    explore_steps: int = 0
    update_steps: int = 0
    total_episodes: int = 0
    details: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        if self.phase == "unknown":
            return "Unknown (no activity detected)"
        elif self.phase == "heatup":
            return f"Heatup ({self.heatup_steps:,} steps)"
        elif self.phase == "training":
            return f"Training ({self.update_steps:,} updates, {self.explore_steps:,} explore)"
        elif self.phase == "evaluation":
            return f"Evaluation (paused {self.activity_age_minutes:.0f}m, likely evaluating)"
        elif self.phase == "completed":
            return "Completed"
        elif self.phase == "crashed":
            return "Crashed/Stopped"
        return self.phase


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)
    warning: bool = False


@dataclass
class ValidationReport:
    """Collection of all validation check results."""
    experiment_name: str
    experiment_path: str
    checks: List[CheckResult] = field(default_factory=list)
    training_status: Optional[TrainingStatus] = None
    docker_status: Optional[str] = None
    
    def add(self, result: CheckResult):
        self.checks.append(result)
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks if not c.warning)
    
    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.warning)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.warning)


# =============================================================================
# Console Colors
# =============================================================================

class Colors:
    """ANSI color codes for console output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.GREEN = ""
        cls.RED = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.BOLD = ""
        cls.RESET = ""


def print_check(result: CheckResult, verbose: bool = False):
    """Print a single check result with color coding."""
    if result.passed:
        status = f"{Colors.GREEN}PASS{Colors.RESET}"
    elif result.warning:
        status = f"{Colors.YELLOW}WARN{Colors.RESET}"
    else:
        status = f"{Colors.RED}FAIL{Colors.RESET}"
    
    print(f"  [{status}] {result.name}: {result.message}")
    
    if verbose and result.details:
        for detail in result.details:
            print(f"         {Colors.BLUE}-{Colors.RESET} {detail}")


# =============================================================================
# Path Resolution
# =============================================================================

DEFAULT_RESULTS_BASE = "results/eve_paper/neurovascular/full/mesh_ben"


def find_experiment_path(experiment_name: str, base_path: Optional[str] = None) -> Optional[Path]:
    """
    Find the experiment folder by name.
    
    Searches for folders containing the experiment name in the results directory.
    """
    if base_path:
        return Path(base_path)
    
    # Try relative path from script location
    script_dir = Path(__file__).parent
    results_dir = script_dir / DEFAULT_RESULTS_BASE
    
    if not results_dir.exists():
        # Try absolute path
        results_dir = Path("D:/stEVE_training/training_scripts") / DEFAULT_RESULTS_BASE
    
    if not results_dir.exists():
        return None
    
    # Find matching folders
    matches = []
    for folder in results_dir.iterdir():
        if folder.is_dir() and experiment_name in folder.name:
            matches.append(folder)
    
    if not matches:
        return None
    
    # Return most recent match if multiple
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


# =============================================================================
# Training Status Detection
# =============================================================================

def get_file_age_minutes(file_path: Path) -> float:
    """Get the age of a file in minutes."""
    if not file_path.exists():
        return float('inf')
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() / 60


def detect_training_status(exp_path: Path) -> TrainingStatus:
    """Detect the current training phase and status."""
    status = TrainingStatus(phase="unknown", is_active=False)
    
    # Check key files for timestamps
    losses_csv = exp_path / "diagnostics" / "csv" / "losses_trainer_synchron.csv"
    main_log = exp_path / "main.log"
    
    # Get most recent activity from any relevant file
    files_to_check = [
        losses_csv,
        main_log,
        exp_path / "logs_subprocesses" / "trainer_synchron.log",
    ]
    
    # Add worker logs
    worker_logs_dir = exp_path / "logs_subprocesses"
    if worker_logs_dir.exists():
        files_to_check.extend(worker_logs_dir.glob("worker_*.log"))
    
    most_recent = None
    for f in files_to_check:
        if f.exists():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if most_recent is None or mtime > most_recent:
                most_recent = mtime
    
    if most_recent:
        status.last_activity = most_recent
        status.activity_age_minutes = (datetime.now() - most_recent).total_seconds() / 60
    
    # Parse main.log for status
    if main_log.exists():
        try:
            with open(main_log, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            
            # Look for heatup/training progress
            for line in reversed(lines[-100:]):  # Check last 100 lines
                # Check for heatup
                if "heatup" in line.lower():
                    match = re.search(r"heatup.*?(\d+(?:\.\d+)?)\s*steps", line, re.IGNORECASE)
                    if match:
                        status.heatup_steps = int(float(match.group(1)))
                        status.details.append(f"Heatup steps configured: {status.heatup_steps:,}")
                
                # Check for update/exploration progress
                if "update / exploration" in line.lower():
                    # Extract update and explore steps
                    update_match = re.search(r"(\d+)\s*steps total", line)
                    if update_match:
                        status.update_steps = int(update_match.group(1))
                    
                    explore_match = re.search(r"/\s*(\d+)\s*steps total", line)
                    if explore_match:
                        status.explore_steps = int(explore_match.group(1))
                    
                    episode_match = re.search(r"(\d+)\s*episodes total", line)
                    if episode_match:
                        status.total_episodes = int(episode_match.group(1))
                    
                    status.details.append(f"Last logged: {status.update_steps:,} updates, {status.explore_steps:,} explore")
                    break
                
                # Check for evaluation
                if "evaluation" in line.lower() and "steps" in line.lower():
                    status.details.append("Evaluation phase detected in logs")
        except Exception as e:
            status.details.append(f"Error parsing main.log: {e}")
    
    # Determine phase based on file states
    losses_age = get_file_age_minutes(losses_csv)
    
    if not losses_csv.exists():
        # No losses yet - likely in heatup or just started
        if main_log.exists() and status.activity_age_minutes < 30:
            status.phase = "heatup"
            status.is_active = True
            status.details.append("No gradient updates yet - likely in heatup phase")
        else:
            status.phase = "unknown"
    elif losses_age < 5:
        # Recent updates - actively training
        status.phase = "training"
        status.is_active = True
        status.details.append(f"Loss CSV updated {losses_age:.1f} min ago - actively training")
    elif losses_age < 120:
        # No updates for a while but not too long - likely in evaluation
        status.phase = "evaluation"
        status.is_active = True
        status.details.append(f"Loss CSV paused for {losses_age:.0f} min - likely in evaluation phase")
    else:
        # Very old - likely completed or crashed
        if status.activity_age_minutes > 120:
            status.phase = "completed"
            status.is_active = False
            status.details.append(f"No activity for {status.activity_age_minutes:.0f} min - likely completed/stopped")
        else:
            status.phase = "evaluation"
            status.is_active = True
    
    return status


def check_docker_container(container_name: str) -> Tuple[str, List[str]]:
    """Check Docker container status and get recent logs."""
    try:
        # Check if container exists and its status
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return "Docker command failed", [result.stderr.strip()]
        
        container_status = result.stdout.strip()
        if not container_status:
            return "Container not found", []
        
        details = [f"Container status: {container_status}"]
        
        # Check if running
        is_running = "Up" in container_status
        
        if is_running:
            # Get recent logs
            result = subprocess.run(
                ["docker", "logs", "--tail", "20", container_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Look for key patterns in logs
                logs = result.stdout + result.stderr
                
                if "heatup" in logs.lower():
                    details.append("Heatup phase detected in Docker logs")
                if "update / exploration" in logs.lower():
                    details.append("Training progress detected in Docker logs")
                if "evaluation" in logs.lower():
                    details.append("Evaluation phase detected in Docker logs")
                if "error" in logs.lower() or "exception" in logs.lower():
                    details.append("Errors detected in Docker logs (check manually)")
                
                # Look for ReplayBuffer status
                for line in logs.split('\n'):
                    if "ReplayBuffer" in line and "STATUS" in line:
                        # Extract episode count
                        match = re.search(r"episodes_received=(\d+)", line)
                        if match:
                            details.append(f"Replay buffer: {match.group(1)} episodes received")
                        match = re.search(r"batches_produced=(\d+)", line)
                        if match:
                            details.append(f"Replay buffer: {match.group(1)} batches produced")
                        break
        
        return container_status, details
        
    except subprocess.TimeoutExpired:
        return "Docker command timed out", []
    except FileNotFoundError:
        return "Docker not available", ["Docker command not found in PATH"]
    except Exception as e:
        return f"Error: {e}", []


# =============================================================================
# Validation Functions
# =============================================================================

def validate_folder_structure(exp_path: Path) -> List[CheckResult]:
    """Validate required folder structure exists."""
    results = []
    
    required_folders = [
        "diagnostics",
        "logs_subprocesses",
        "checkpoints",
        "config",
    ]
    
    for folder in required_folders:
        folder_path = exp_path / folder
        if folder_path.exists() and folder_path.is_dir():
            results.append(CheckResult(
                name=f"Folder: {folder}/",
                passed=True,
                message="exists"
            ))
        else:
            results.append(CheckResult(
                name=f"Folder: {folder}/",
                passed=False,
                message="MISSING"
            ))
    
    # Check diagnostics subfolders
    diag_subfolders = ["csv", "probes", "policy_snapshots", "logs_subprocesses"]
    for subfolder in diag_subfolders:
        subfolder_path = exp_path / "diagnostics" / subfolder
        if subfolder_path.exists():
            results.append(CheckResult(
                name=f"Folder: diagnostics/{subfolder}/",
                passed=True,
                message="exists"
            ))
        else:
            results.append(CheckResult(
                name=f"Folder: diagnostics/{subfolder}/",
                passed=False,
                message="MISSING",
                warning=True
            ))
    
    return results


def validate_losses_csv(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate the losses CSV file."""
    csv_path = exp_path / "diagnostics" / "csv" / "losses_trainer_synchron.csv"
    
    if not csv_path.exists():
        # Provide context based on training status
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Losses CSV",
                passed=True,
                message="Not created yet (expected during heatup)",
                warning=True,
                details=["Gradient updates start after heatup completes"]
            )
        return CheckResult(
            name="Losses CSV",
            passed=False,
            message="File not found"
        )
    
    try:
        file_size = csv_path.stat().st_size
        if file_size == 0:
            return CheckResult(
                name="Losses CSV",
                passed=False,
                message="File is empty (0 bytes)"
            )
        
        # Read and validate CSV
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            
            # Check required columns
            required_cols = ["update_step", "explore_step", "q1_loss", "q2_loss", "policy_loss", "alpha"]
            v3_cols = ["wall_time"]  # New in v3
            missing_cols = [c for c in required_cols if c not in headers]
            missing_v3_cols = [c for c in v3_cols if c not in headers]
            
            if missing_cols:
                return CheckResult(
                    name="Losses CSV",
                    passed=False,
                    message=f"Missing columns: {missing_cols}"
                )
            
            has_v3_format = len(missing_v3_cols) == 0
            
            # Count rows and check for issues
            row_count = 0
            nan_count = 0
            inf_count = 0
            last_update_step = 0
            gaps = []
            
            for row in reader:
                row_count += 1
                
                update_step = int(row.get("update_step", 0))
                if update_step < last_update_step:
                    gaps.append(f"Non-monotonic at row {row_count}")
                last_update_step = update_step
                
                # Check for NaN/Inf in numeric columns
                for col in ["q1_loss", "q2_loss", "policy_loss"]:
                    val = row.get(col, "")
                    if val.lower() == "nan":
                        nan_count += 1
                    elif val.lower() == "inf" or val.lower() == "-inf":
                        inf_count += 1
        
        # Get file age for activity indicator
        file_age = get_file_age_minutes(csv_path)
        age_str = f"{file_age:.0f}m ago" if file_age < 60 else f"{file_age/60:.1f}h ago"
        
        format_version = "v3" if has_v3_format else "v2"
        details = [
            f"Format: {format_version}" + (" (has wall_time)" if has_v3_format else " (no wall_time)"),
            f"Rows: {row_count:,}",
            f"Last update_step: {last_update_step:,}",
            f"File size: {file_size:,} bytes",
            f"Last modified: {age_str}"
        ]
        
        if nan_count > 0:
            details.append(f"NaN values: {nan_count}")
        if inf_count > 0:
            details.append(f"Inf values: {inf_count}")
        if gaps:
            details.extend(gaps[:5])  # Show first 5 gaps
        
        # Determine status
        if nan_count > row_count * 0.1 or inf_count > row_count * 0.1:
            return CheckResult(
                name="Losses CSV",
                passed=False,
                message=f"Too many invalid values ({nan_count} NaN, {inf_count} Inf)",
                details=details
            )
        
        if gaps:
            return CheckResult(
                name="Losses CSV",
                passed=True,
                message=f"{row_count:,} rows, {len(gaps)} gaps",
                details=details,
                warning=True
            )
        
        return CheckResult(
            name="Losses CSV",
            passed=True,
            message=f"{row_count:,} rows logged",
            details=details
        )
        
    except Exception as e:
        return CheckResult(
            name="Losses CSV",
            passed=False,
            message=f"Error reading file: {e}"
        )


def validate_batch_samples(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate the batch samples JSONL file."""
    jsonl_path = exp_path / "diagnostics" / "csv" / "batch_samples_trainer_synchron.jsonl"
    
    if not jsonl_path.exists():
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Batch Samples JSONL",
                passed=True,
                message="Not created yet (expected during heatup)",
                warning=True,
                details=["Batch sampling starts after gradient updates begin"]
            )
        return CheckResult(
            name="Batch Samples JSONL",
            passed=False,
            message="File not found"
        )
    
    try:
        file_size = jsonl_path.stat().st_size
        if file_size == 0:
            return CheckResult(
                name="Batch Samples JSONL",
                passed=False,
                message="File is empty (0 bytes)"
            )
        
        # Read and validate JSONL
        line_count = 0
        valid_samples = 0
        invalid_samples = 0
        required_keys = ["update_step", "explore_step", "n_samples", "samples"]
        sample_keys = ["state", "reward"]  # action_taken or action
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                try:
                    entry = json.loads(line.strip())
                    
                    # Check required keys
                    if all(k in entry for k in required_keys):
                        # Check sample structure
                        samples = entry.get("samples", [])
                        if samples and all(k in samples[0] for k in sample_keys):
                            valid_samples += 1
                        else:
                            invalid_samples += 1
                    else:
                        invalid_samples += 1
                        
                except json.JSONDecodeError:
                    invalid_samples += 1
        
        details = [
            f"Lines: {line_count:,}",
            f"Valid entries: {valid_samples:,}",
            f"File size: {file_size:,} bytes"
        ]
        
        if invalid_samples > 0:
            details.append(f"Invalid entries: {invalid_samples}")
        
        if invalid_samples > line_count * 0.1:
            return CheckResult(
                name="Batch Samples JSONL",
                passed=False,
                message=f"Too many invalid entries ({invalid_samples}/{line_count})",
                details=details
            )
        
        return CheckResult(
            name="Batch Samples JSONL",
            passed=True,
            message=f"{valid_samples:,} entries logged",
            details=details
        )
        
    except Exception as e:
        return CheckResult(
            name="Batch Samples JSONL",
            passed=False,
            message=f"Error reading file: {e}"
        )


def validate_probe_values(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate the probe values JSONL file."""
    jsonl_path = exp_path / "diagnostics" / "csv" / "probe_values_trainer_synchron.jsonl"
    
    if not jsonl_path.exists():
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Probe Values JSONL",
                passed=True,
                message="Not created yet (expected during heatup)",
                warning=True,
                details=["Probe values logged after gradient updates begin"]
            )
        return CheckResult(
            name="Probe Values JSONL",
            passed=False,
            message="File not found"
        )
    
    try:
        file_size = jsonl_path.stat().st_size
        if file_size == 0:
            return CheckResult(
                name="Probe Values JSONL",
                passed=False,
                message="File is empty (0 bytes)"
            )
        
        # Count lines (quick check)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        
        details = [
            f"Lines: {line_count:,}",
            f"File size: {file_size:,} bytes"
        ]
        
        return CheckResult(
            name="Probe Values JSONL",
            passed=True,
            message=f"{line_count:,} entries logged",
            details=details
        )
        
    except Exception as e:
        return CheckResult(
            name="Probe Values JSONL",
            passed=False,
            message=f"Error reading file: {e}"
        )


def validate_episode_summary(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate the episode_summary.jsonl file (new in v3)."""
    jsonl_path = exp_path / "diagnostics" / "csv" / "episode_summary.jsonl"
    
    if not jsonl_path.exists():
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Episode Summary JSONL",
                passed=True,
                message="Not created yet (expected during heatup)",
                warning=True,
                details=["Episode summaries logged after training begins"]
            )
        return CheckResult(
            name="Episode Summary JSONL",
            passed=False,
            message="File not found (required for v3 diagnostics)"
        )
    
    try:
        file_size = jsonl_path.stat().st_size
        if file_size == 0:
            return CheckResult(
                name="Episode Summary JSONL",
                passed=False,
                message="File is empty (0 bytes)"
            )
        
        # Read and validate JSONL
        line_count = 0
        valid_entries = 0
        has_wall_time = 0
        has_explore_step = 0
        has_update_step = 0
        required_keys = ["wall_time", "explore_step", "update_step", "episode_id"]
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                try:
                    entry = json.loads(line.strip())
                    
                    if all(k in entry for k in required_keys):
                        valid_entries += 1
                        if entry.get("wall_time"):
                            has_wall_time += 1
                        if entry.get("explore_step"):
                            has_explore_step += 1
                        if entry.get("update_step"):
                            has_update_step += 1
                except json.JSONDecodeError:
                    pass
        
        details = [
            f"Lines: {line_count:,}",
            f"Valid entries: {valid_entries:,}",
            f"With wall_time: {has_wall_time:,}",
            f"File size: {file_size:,} bytes"
        ]
        
        if valid_entries == 0:
            return CheckResult(
                name="Episode Summary JSONL",
                passed=False,
                message="No valid entries found",
                details=details
            )
        
        return CheckResult(
            name="Episode Summary JSONL",
            passed=True,
            message=f"{valid_entries:,} episodes logged",
            details=details
        )
        
    except Exception as e:
        return CheckResult(
            name="Episode Summary JSONL",
            passed=False,
            message=f"Error reading file: {e}"
        )


def validate_probe_states(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate the probe states NPZ file."""
    npz_path = exp_path / "diagnostics" / "probes" / "probe_states.npz"
    
    if not npz_path.exists():
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Probe States NPZ",
                passed=True,
                message="Not created yet (expected during heatup)",
                warning=True,
                details=["Probe states captured after heatup completes"]
            )
        return CheckResult(
            name="Probe States NPZ",
            passed=False,
            message="File not found"
        )
    
    if not HAS_NUMPY:
        file_size = npz_path.stat().st_size
        return CheckResult(
            name="Probe States NPZ",
            passed=True,
            message=f"File exists ({file_size} bytes), numpy not available for full validation",
            warning=True
        )
    
    try:
        data = np.load(npz_path)
        
        if "probe_states" not in data:
            return CheckResult(
                name="Probe States NPZ",
                passed=False,
                message="Missing 'probe_states' key"
            )
        
        probe_states = data["probe_states"]
        n_states, state_dim = probe_states.shape
        file_size = npz_path.stat().st_size
        
        details = [
            f"Number of probe states: {n_states}",
            f"State dimension: {state_dim}",
            f"File size: {file_size} bytes"
        ]
        
        return CheckResult(
            name="Probe States NPZ",
            passed=True,
            message=f"{n_states} probe states (dim={state_dim})",
            details=details
        )
        
    except Exception as e:
        return CheckResult(
            name="Probe States NPZ",
            passed=False,
            message=f"Error loading file: {e}"
        )


def validate_policy_snapshots(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate policy snapshot files."""
    snapshots_dir = exp_path / "diagnostics" / "policy_snapshots"
    
    if not snapshots_dir.exists():
        if training_status and training_status.phase in ("heatup", "training") and training_status.update_steps < 10000:
            return CheckResult(
                name="Policy Snapshots",
                passed=True,
                message="Folder not created yet (snapshots start at 10k updates)",
                warning=True
            )
        return CheckResult(
            name="Policy Snapshots",
            passed=False,
            message="Folder not found"
        )
    
    snapshots = list(snapshots_dir.glob("policy_*.pt"))
    
    if not snapshots:
        if training_status and training_status.update_steps < 10000:
            return CheckResult(
                name="Policy Snapshots",
                passed=True,
                message=f"No snapshots yet ({training_status.update_steps:,} updates, first at 10k)",
                warning=True
            )
        return CheckResult(
            name="Policy Snapshots",
            passed=False,
            message="No policy_*.pt files found"
        )
    
    # Get file sizes and steps
    snapshot_info = []
    total_size = 0
    for snap in snapshots:
        match = re.search(r"policy_(\d+)\.pt", snap.name)
        step = int(match.group(1)) if match else 0
        size = snap.stat().st_size
        total_size += size
        snapshot_info.append((step, size, snap.name))
    
    snapshot_info.sort(key=lambda x: x[0])
    
    details = [f"{name}: step {step:,}, {size:,} bytes" for step, size, name in snapshot_info]
    
    # Check for reasonable file sizes (should be ~15MB each for this model)
    min_size = min(s[1] for s in snapshot_info)
    if min_size < 1000:
        return CheckResult(
            name="Policy Snapshots",
            passed=False,
            message=f"Some snapshots are too small ({min_size} bytes)",
            details=details,
            warning=True
        )
    
    return CheckResult(
        name="Policy Snapshots",
        passed=True,
        message=f"{len(snapshots)} snapshots ({total_size / 1e6:.1f} MB total)",
        details=details
    )


def validate_worker_logs(exp_path: Path, verbose: bool = False, training_status: Optional[TrainingStatus] = None) -> CheckResult:
    """Validate worker log files.
    
    Checks both old location (logs_subprocesses/) and new location (diagnostics/logs_subprocesses/).
    Validates v3 format with full timestamps and wall_time/pid fields.
    """
    # Check both old and new locations for worker logs
    logs_dir_old = exp_path / "logs_subprocesses"
    logs_dir_new = exp_path / "diagnostics" / "logs_subprocesses"
    
    logs_dir = None
    location_str = ""
    
    if logs_dir_new.exists():
        logs_dir = logs_dir_new
        location_str = "diagnostics/logs_subprocesses/"
    elif logs_dir_old.exists():
        logs_dir = logs_dir_old
        location_str = "logs_subprocesses/"
    
    if logs_dir is None:
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Worker Logs",
                passed=True,
                message="Not created yet (expected during heatup)",
                warning=True,
                details=["Worker logs created after environment steps begin"]
            )
        return CheckResult(
            name="Worker Logs",
            passed=False,
            message="No worker logs folder found (checked logs_subprocesses/ and diagnostics/logs_subprocesses/)"
        )
    
    worker_logs = list(logs_dir.glob("worker_*.log"))
    trainer_log = logs_dir.parent / "logs_subprocesses" / "trainer_synchron.log" if "diagnostics" in str(logs_dir) else logs_dir / "trainer_synchron.log"
    
    # Also check old location for trainer log
    if not trainer_log.exists():
        trainer_log = exp_path / "logs_subprocesses" / "trainer_synchron.log"
    
    details = [f"Location: {location_str}"]
    
    # Check trainer log
    if trainer_log.exists():
        trainer_size = trainer_log.stat().st_size
        details.append(f"trainer_synchron.log: {trainer_size:,} bytes")
    else:
        details.append("trainer_synchron.log: not found")
    
    # Check worker logs
    if not worker_logs:
        if training_status and training_status.phase == "heatup":
            return CheckResult(
                name="Worker Logs",
                passed=True,
                message="No worker logs yet (expected during early heatup)",
                warning=True,
                details=details
            )
        return CheckResult(
            name="Worker Logs",
            passed=False,
            message="No worker_*.log files found",
            details=details
        )
    
    # Worker logs now use PID instead of sequential numbers
    # Format: worker_{pid}.log
    n_workers = len(worker_logs)
    total_size = sum(log.stat().st_size for log in worker_logs)
    details.append(f"Worker log files: {n_workers}")
    details.append(f"Total worker logs: {total_size:,} bytes")
    
    # Check for empty logs
    empty_logs = [log.name for log in worker_logs if log.stat().st_size == 0]
    if empty_logs:
        details.append(f"Empty logs: {len(empty_logs)}")
    
    # Validate v3 format in one of the worker logs
    v3_format_ok = False
    has_full_timestamp = False
    has_wall_time = False
    has_pid = False
    format_issues = []
    
    sample_log = max(worker_logs, key=lambda p: p.stat().st_size)  # Check largest log
    try:
        with open(sample_log, "r", encoding="utf-8", errors="replace") as f:
            lines_checked = 0
            for line in f:
                if lines_checked > 100:  # Check first 100 lines
                    break
                lines_checked += 1
                
                # Check for full timestamp format (YYYY-MM-DD HH:MM:SS)
                if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
                    has_full_timestamp = True
                
                # Check for wall_time field
                if "wall_time=" in line:
                    has_wall_time = True
                
                # Check for pid field
                if "pid=" in line:
                    has_pid = True
                
                # If we found all v3 indicators, we're good
                if has_full_timestamp and has_wall_time and has_pid:
                    v3_format_ok = True
                    break
        
        if has_full_timestamp:
            details.append("Timestamp format: v3 (full datetime)")
        else:
            details.append("Timestamp format: v2 (time only)")
            format_issues.append("Missing full date in timestamps")
        
        if has_wall_time:
            details.append("wall_time field: present")
        else:
            format_issues.append("Missing wall_time field")
        
        if has_pid:
            details.append("pid field: present")
        else:
            format_issues.append("Missing pid field")
        
    except Exception as e:
        details.append(f"Format check error: {e}")
    
    # Determine result
    if empty_logs:
        return CheckResult(
            name="Worker Logs",
            passed=True,
            message=f"{n_workers} workers, {len(empty_logs)} empty",
            details=details,
            warning=True
        )
    
    if format_issues:
        return CheckResult(
            name="Worker Logs",
            passed=True,
            message=f"{n_workers} workers (v2 format)",
            details=details,
            warning=True
        )
    
    return CheckResult(
        name="Worker Logs",
        passed=True,
        message=f"{n_workers} workers logged (v3 format)",
        details=details
    )


def validate_main_log(exp_path: Path, verbose: bool = False) -> CheckResult:
    """Validate main.log file."""
    main_log = exp_path / "main.log"
    
    if not main_log.exists():
        return CheckResult(
            name="Main Log",
            passed=False,
            message="File not found"
        )
    
    try:
        file_size = main_log.stat().st_size
        
        if file_size == 0:
            return CheckResult(
                name="Main Log",
                passed=False,
                message="File is empty"
            )
        
        # Read and analyze log
        error_count = 0
        warning_count = 0
        training_entries = 0
        
        with open(main_log, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_lower = line.lower()
                if "error" in line_lower:
                    error_count += 1
                if "warning" in line_lower:
                    warning_count += 1
                if "update / exploration" in line_lower or "heatup" in line_lower:
                    training_entries += 1
        
        details = [
            f"File size: {file_size:,} bytes",
            f"Training progress entries: {training_entries}",
            f"Errors: {error_count}",
            f"Warnings: {warning_count}"
        ]
        
        if error_count > 0:
            return CheckResult(
                name="Main Log",
                passed=True,
                message=f"{error_count} errors found",
                details=details,
                warning=True
            )
        
        return CheckResult(
            name="Main Log",
            passed=True,
            message=f"Valid ({training_entries} progress entries)",
            details=details
        )
        
    except Exception as e:
        return CheckResult(
            name="Main Log",
            passed=False,
            message=f"Error reading file: {e}"
        )


def validate_config_files(exp_path: Path, verbose: bool = False) -> List[CheckResult]:
    """Validate config files."""
    results = []
    
    config_files = [
        ("env_train.yml", "Training environment config"),
        ("env_eval.yml", "Evaluation environment config"),
        ("runner.yml", "Runner config"),
    ]
    
    for filename, desc in config_files:
        file_path = exp_path / filename
        
        if not file_path.exists():
            results.append(CheckResult(
                name=f"Config: {filename}",
                passed=False,
                message="File not found"
            ))
            continue
        
        file_size = file_path.stat().st_size
        
        if HAS_YAML:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
                results.append(CheckResult(
                    name=f"Config: {filename}",
                    passed=True,
                    message=f"Valid YAML ({file_size} bytes)"
                ))
            except Exception as e:
                results.append(CheckResult(
                    name=f"Config: {filename}",
                    passed=False,
                    message=f"Invalid YAML: {e}"
                ))
        else:
            results.append(CheckResult(
                name=f"Config: {filename}",
                passed=True,
                message=f"Exists ({file_size} bytes)",
                warning=True
            ))
    
    return results


def validate_checkpoints(exp_path: Path, verbose: bool = False) -> CheckResult:
    """Validate checkpoint files."""
    checkpoints_dir = exp_path / "checkpoints"
    
    if not checkpoints_dir.exists():
        return CheckResult(
            name="Checkpoints",
            passed=False,
            message="checkpoints/ folder not found"
        )
    
    checkpoints = list(checkpoints_dir.glob("*.everl"))
    
    if not checkpoints:
        return CheckResult(
            name="Checkpoints",
            passed=False,
            message="No checkpoint files found"
        )
    
    # Check for best_checkpoint
    best_checkpoint = checkpoints_dir / "best_checkpoint.everl"
    has_best = best_checkpoint.exists()
    
    details = []
    total_size = 0
    
    for cp in checkpoints:
        size = cp.stat().st_size
        total_size += size
        details.append(f"{cp.name}: {size:,} bytes")
    
    if not has_best:
        details.insert(0, "WARNING: best_checkpoint.everl missing")
    
    if not has_best:
        return CheckResult(
            name="Checkpoints",
            passed=True,
            message=f"{len(checkpoints)} checkpoints, no best_checkpoint",
            details=details,
            warning=True
        )
    
    return CheckResult(
        name="Checkpoints",
        passed=True,
        message=f"{len(checkpoints)} checkpoints ({total_size / 1e6:.1f} MB)",
        details=details
    )


# =============================================================================
# Main Validation
# =============================================================================

def validate_experiment(experiment_name: str, full_path: Optional[str] = None, verbose: bool = False, docker_container: Optional[str] = None) -> ValidationReport:
    """Run all validation checks for an experiment."""
    
    # Find experiment path
    if full_path:
        exp_path = Path(full_path)
    else:
        exp_path = find_experiment_path(experiment_name)
    
    if not exp_path or not exp_path.exists():
        report = ValidationReport(
            experiment_name=experiment_name,
            experiment_path=str(exp_path) if exp_path else "NOT FOUND"
        )
        report.add(CheckResult(
            name="Experiment Path",
            passed=False,
            message=f"Could not find experiment folder for '{experiment_name}'"
        ))
        return report
    
    report = ValidationReport(
        experiment_name=experiment_name,
        experiment_path=str(exp_path)
    )
    
    # Detect training status
    training_status = detect_training_status(exp_path)
    report.training_status = training_status
    
    # Check Docker container if specified
    if docker_container:
        docker_status, docker_details = check_docker_container(docker_container)
        report.docker_status = docker_status
        training_status.details.extend(docker_details)
    
    # Run all checks
    print(f"\n{Colors.BOLD}Validating: {exp_path}{Colors.RESET}\n")
    
    # Print training status first
    print(f"{Colors.BOLD}Training Status:{Colors.RESET}")
    if training_status.is_active:
        status_color = Colors.GREEN
    else:
        status_color = Colors.YELLOW if training_status.phase == "completed" else Colors.RED
    print(f"  Phase: {status_color}{training_status.summary()}{Colors.RESET}")
    
    if training_status.last_activity:
        age_str = f"{training_status.activity_age_minutes:.0f}m ago" if training_status.activity_age_minutes < 60 else f"{training_status.activity_age_minutes/60:.1f}h ago"
        print(f"  Last activity: {age_str}")
    
    if verbose and training_status.details:
        for detail in training_status.details:
            print(f"    {Colors.BLUE}-{Colors.RESET} {detail}")
    
    if docker_container:
        print(f"\n{Colors.BOLD}Docker Container: {docker_container}{Colors.RESET}")
        print(f"  Status: {report.docker_status}")
    
    print()
    
    # 1. Folder structure
    print(f"{Colors.BOLD}1. Folder Structure{Colors.RESET}")
    for result in validate_folder_structure(exp_path):
        report.add(result)
        print_check(result, verbose)
    
    # 2. Diagnostics files
    print(f"\n{Colors.BOLD}2. Diagnostics Files{Colors.RESET}")
    
    result = validate_losses_csv(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    result = validate_batch_samples(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    result = validate_probe_values(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    result = validate_probe_states(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    result = validate_policy_snapshots(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    result = validate_episode_summary(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    # 3. Worker logs
    print(f"\n{Colors.BOLD}3. Worker Logs{Colors.RESET}")
    result = validate_worker_logs(exp_path, verbose, training_status)
    report.add(result)
    print_check(result, verbose)
    
    # 4. Main logs
    print(f"\n{Colors.BOLD}4. Main Logs{Colors.RESET}")
    result = validate_main_log(exp_path, verbose)
    report.add(result)
    print_check(result, verbose)
    
    for result in validate_config_files(exp_path, verbose):
        report.add(result)
        print_check(result, verbose)
    
    # 5. Checkpoints
    print(f"\n{Colors.BOLD}5. Checkpoints{Colors.RESET}")
    result = validate_checkpoints(exp_path, verbose)
    report.add(result)
    print_check(result, verbose)
    
    return report


def print_summary(report: ValidationReport):
    """Print validation summary."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"Experiment: {report.experiment_name}")
    print(f"Path: {report.experiment_path}")
    
    if report.training_status:
        ts = report.training_status
        if ts.is_active:
            status_str = f"{Colors.GREEN}ACTIVE{Colors.RESET}"
        else:
            status_str = f"{Colors.YELLOW}INACTIVE{Colors.RESET}"
        print(f"Status: {status_str} - {ts.summary()}")
    
    if report.docker_status:
        print(f"Docker: {report.docker_status}")
    
    print()
    
    if report.all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}ALL CHECKS PASSED{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}SOME CHECKS FAILED{Colors.RESET}")
    
    print(f"\n  {Colors.GREEN}Passed:{Colors.RESET}   {report.pass_count}")
    print(f"  {Colors.RED}Failed:{Colors.RESET}   {report.fail_count}")
    print(f"  {Colors.YELLOW}Warnings:{Colors.RESET} {report.warning_count}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate experiment log files and diagnostics"
    )
    parser.add_argument(
        "experiment_name",
        nargs="?",
        help="Name of the experiment (e.g., diag_debug_test5)"
    )
    parser.add_argument(
        "--full-path",
        type=str,
        help="Full path to experiment folder"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information for each check"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--docker-container", "-d",
        type=str,
        help="Docker container name to check logs (e.g., diag_debug_test5)"
    )
    
    args = parser.parse_args()
    
    if not args.experiment_name and not args.full_path:
        parser.error("Either experiment_name or --full-path is required")
    
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    experiment_name = args.experiment_name or os.path.basename(args.full_path)
    
    report = validate_experiment(
        experiment_name=experiment_name,
        full_path=args.full_path,
        verbose=args.verbose,
        docker_container=args.docker_container
    )
    
    print_summary(report)
    
    # Exit with appropriate code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
