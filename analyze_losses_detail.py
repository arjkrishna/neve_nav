import csv
import statistics

loss_file = r'training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-02-03_003538_diag_debug_test4\diagnostics\csv\losses_trainer_synchron.csv'

with open(loss_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total updates: {len(rows)}')
print()

# Divide into 10 segments
seg_size = len(rows) // 10

print("=" * 100)
print("COMPREHENSIVE TRAINING ANALYSIS")
print("=" * 100)
print()

print("SEGMENT | Steps K | Q1 Loss | Q2 Loss | Pol Loss | Alpha | Q1 Mean | Entropy | Grad Q1 | Grad Pol")
print("-" * 100)

for i in range(10):
    start = i * seg_size
    end = len(rows) - 1 if i == 9 else (i + 1) * seg_size - 1
    seg = rows[start:end+1]
    
    exp_start = int(seg[0]['explore_step']) // 1000
    exp_end = int(seg[-1]['explore_step']) // 1000
    
    q1_loss = statistics.mean(float(r['q1_loss']) for r in seg)
    q2_loss = statistics.mean(float(r['q2_loss']) for r in seg)
    pol_loss = statistics.mean(float(r['policy_loss']) for r in seg)
    alpha = statistics.mean(float(r['alpha']) for r in seg)
    q1_mean = statistics.mean(float(r['q1_mean']) for r in seg)
    entropy = statistics.mean(float(r['entropy_proxy']) for r in seg)
    grad_q1 = statistics.mean(float(r['grad_norm_q1']) for r in seg)
    grad_pol = statistics.mean(float(r['grad_norm_policy']) for r in seg)
    
    print(f"  {i+1:2}    | {exp_start:3}-{exp_end:3} | {q1_loss:7.4f} | {q2_loss:7.4f} | {pol_loss:8.2f} | {alpha:6.4f} | {q1_mean:7.2f} | {entropy:7.2f} | {grad_q1:7.2f} | {grad_pol:8.2f}")

print()
print("KEY OBSERVATIONS:")
print("-" * 50)

# Get first and last segment stats
first_seg = rows[:seg_size]
last_seg = rows[-seg_size:]

first_q = statistics.mean(float(r['q1_mean']) for r in first_seg)
last_q = statistics.mean(float(r['q1_mean']) for r in last_seg)
q_change = (last_q - first_q) / first_q * 100 if first_q != 0 else 0

first_alpha = statistics.mean(float(r['alpha']) for r in first_seg)
last_alpha = statistics.mean(float(r['alpha']) for r in last_seg)
alpha_change = (last_alpha - first_alpha) / first_alpha * 100 if first_alpha != 0 else 0

first_entropy = statistics.mean(float(r['entropy_proxy']) for r in first_seg)
last_entropy = statistics.mean(float(r['entropy_proxy']) for r in last_seg)

print(f"Q-value change: {first_q:.2f} -> {last_q:.2f} ({q_change:+.1f}%)")
print(f"Alpha change: {first_alpha:.4f} -> {last_alpha:.6f} ({alpha_change:+.1f}%)")
print(f"Entropy change: {first_entropy:.2f} -> {last_entropy:.2f}")
print()

# Check for collapse indicators
if last_q < first_q * 0.5:
    print("WARNING: Q-values dropped by more than 50% - POLICY COLLAPSE DETECTED")
if last_alpha < 0.001:
    print("WARNING: Alpha near zero - EXPLORATION COLLAPSED")
if last_entropy < 0:
    print("WARNING: Negative entropy - EXTREME DETERMINISM")
