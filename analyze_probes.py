import json
import math

probe_file = r'training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-02-03_003538_diag_debug_test4\diagnostics\csv\probe_values_trainer_synchron.jsonl'

with open(probe_file, 'r') as f:
    lines = f.readlines()

print(f'Total probe entries: {len(lines)}')
print()

# Sample at beginning, middle, end
indices = [0, len(lines)//4, len(lines)//2, 3*len(lines)//4, len(lines)-1]

print('Update | Avg Q1 | Avg Q2 | Trans Mean | Trans Std')
print('-' * 60)

for idx in indices:
    entry = json.loads(lines[idx])
    
    # q1/q2: list of [[[scalar]]] per probe
    q1_avg = sum(q[0][0] for q in entry['q1']) / len(entry['q1'])
    q2_avg = sum(q[0][0] for q in entry['q2']) / len(entry['q2'])
    
    # policy_mean: list of [[4 values]] per probe - first is translation
    trans_mean = sum(a[0][0] for a in entry['policy_mean']) / len(entry['policy_mean'])
    
    # policy_log_std: list of [[4 values]] per probe
    trans_std = sum(math.exp(s[0][0]) for s in entry['policy_log_std']) / len(entry['policy_log_std'])
    
    print(f'{entry["update_step"]:6} | {q1_avg:6.2f} | {q2_avg:6.2f} | {trans_mean:10.4f} | {trans_std:9.5f}')

print()
print("=== DETAILED EVOLUTION (every 50 entries) ===")
print()
print('Update | Avg Q1 | Avg Q2 | Trans Mean | Trans Std | Log Std')
print('-' * 70)

for i in range(0, len(lines), 50):
    entry = json.loads(lines[i])
    
    q1_avg = sum(q[0][0] for q in entry['q1']) / len(entry['q1'])
    q2_avg = sum(q[0][0] for q in entry['q2']) / len(entry['q2'])
    trans_mean = sum(a[0][0] for a in entry['policy_mean']) / len(entry['policy_mean'])
    trans_log_std = sum(s[0][0] for s in entry['policy_log_std']) / len(entry['policy_log_std'])
    trans_std = math.exp(trans_log_std)
    
    print(f'{entry["update_step"]:6} | {q1_avg:6.2f} | {q2_avg:6.2f} | {trans_mean:10.4f} | {trans_std:9.5f} | {trans_log_std:7.3f}')
