"""Detailed analysis of diag_debug_test5 run."""
import csv
import json
import statistics
from pathlib import Path

run_dir = Path(r"D:\stEVE_training\training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-02-04_205131_diag_debug_test5")

# Load losses
losses_file = run_dir / "diagnostics/csv/losses_trainer_synchron.csv"
with open(losses_file, 'r') as f:
    reader = csv.DictReader(f)
    losses = list(reader)

print(f"Total updates: {len(losses)}")
print(f"Explore steps: {losses[0]['explore_step']} -> {losses[-1]['explore_step']}")
print()

# Key events from the report
print("=" * 80)
print("EVENT TIMELINE")
print("=" * 80)
print()
print("1. Critic Loss Spike (update=1, explore=95k)")
print("   - Q losses exceed 8× baseline immediately after heatup")
print()
print("2. Actor Actions Negative (update=7200, explore=199k)")
print("   - Policy starts outputting negative (retract) actions")
print()
print("3. Replay Actions Negative (update=10100, explore=278k)")
print("   - Buffer filled with retract behavior")
print()
print("4. Alpha Shutdown (update=20929, explore=485k)")
print("   - Entropy temperature collapses")
print()
print("5. Entropy Drop (update=32930, explore=696k)")
print("   - Policy becomes deterministic")
print()
print("6. Policy Gradient Collapse (update=60088, explore=1259k)")
print("   - Learning stops")
print()

# Analyze loss progression at key points
key_updates = [1, 1000, 7200, 10100, 20929, 32930, 60088, 70000]
print("=" * 80)
print("LOSSES AT KEY POINTS")
print("=" * 80)
print()
print("Update | Explore  | Q1 Loss | Q2 Loss | Pol Loss | Alpha  | Q1 Mean | Entropy")
print("-" * 85)

for update in key_updates:
    # Find closest entry
    closest = min(range(len(losses)), key=lambda i: abs(int(losses[i]['update_step']) - update))
    row = losses[closest]
    print(f"{row['update_step']:>6} | {int(float(row['explore_step']))//1000:>6}k | {float(row['q1_loss']):7.4f} | {float(row['q2_loss']):7.4f} | {float(row['policy_loss']):8.2f} | {float(row['alpha']):6.4f} | {float(row['q1_mean']):7.2f} | {float(row['entropy_proxy']):7.2f}")

# Analyze by segments
print()
print("=" * 80)
print("TRAINING PROGRESSION (10 SEGMENTS)")
print("=" * 80)
print()

seg_size = len(losses) // 10
print("Seg | Updates      | Q1 Loss | Alpha  | Q1 Mean | Entropy | Grad Q1 | Grad Pol")
print("-" * 85)

for i in range(10):
    start = i * seg_size
    end = len(losses) - 1 if i == 9 else (i + 1) * seg_size - 1
    seg = losses[start:end+1]
    
    up_start = int(seg[0]['update_step'])
    up_end = int(seg[-1]['update_step'])
    
    q1_loss_avg = statistics.mean(float(r['q1_loss']) for r in seg)
    alpha_avg = statistics.mean(float(r['alpha']) for r in seg)
    q1_mean_avg = statistics.mean(float(r['q1_mean']) for r in seg)
    entropy_avg = statistics.mean(float(r['entropy_proxy']) for r in seg)
    grad_q1_avg = statistics.mean(float(r['grad_norm_q1']) for r in seg)
    grad_pol_avg = statistics.mean(float(r['grad_norm_policy']) for r in seg)
    
    print(f" {i+1:2} | {up_start:6}-{up_end:6} | {q1_loss_avg:7.4f} | {alpha_avg:6.4f} | {q1_mean_avg:7.2f} | {entropy_avg:7.2f} | {grad_q1_avg:7.2f} | {grad_pol_avg:8.2f}")

# Check probe values
probe_file = run_dir / "diagnostics/csv/probe_values_trainer_synchron.jsonl"
with open(probe_file, 'r') as f:
    probe_lines = f.readlines()

print()
print("=" * 80)
print("PROBE ANALYSIS (Start State Actions)")
print("=" * 80)
print()

# Sample probe entries
indices = [0, len(probe_lines)//4, len(probe_lines)//2, 3*len(probe_lines)//4, len(probe_lines)-1]
print("Update | Q1 Avg | Q2 Avg | Trans Mean")
print("-" * 45)

for idx in indices:
    entry = json.loads(probe_lines[idx])
    
    # Extract Q values (nested lists)
    q1_vals = [q[0][0] for q in entry['q1']]
    q2_vals = [q[0][0] for q in entry['q2']]
    
    # Extract translation means
    trans_means = [a[0][0] for a in entry['policy_mean']]
    
    q1_avg = statistics.mean(q1_vals)
    q2_avg = statistics.mean(q2_vals)
    trans_avg = statistics.mean(trans_means)
    
    print(f"{entry['update_step']:6} | {q1_avg:6.2f} | {q2_avg:6.2f} | {trans_avg:10.4f}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("This run shows CRITIC INSTABILITY as the root cause:")
print()
print("1. Critic losses spike IMMEDIATELY after heatup (update=1)")
print("2. This precedes policy degradation by ~7000 updates")
print("3. Policy responds by learning retract/freeze (negative actions)")
print("4. Alpha and entropy collapse as consequence, not cause")
print()
print("Key difference from test4:")
print("  - test4: Policy found local optimum → α collapsed → stuck")
print("  - test5: Critic instability → policy learned wrong values → collapsed")
print()
print("This suggests a problem with:")
print("  - Critic network initialization or architecture")
print("  - Replay buffer data quality after heatup")
print("  - Reward scaling or target Q computation")
