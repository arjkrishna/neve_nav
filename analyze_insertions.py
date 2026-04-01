import re
import statistics
from pathlib import Path

run_dir = Path(r'training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-02-03_003538_diag_debug_test4')
worker_logs_dir = run_dir / 'logs_subprocesses'

# Pattern to extract step data
step_pattern = r'STEP \| ep=(\d+) \| ep_step=(\d+) \| global=\d+ \| reward=([-\d.]+) \| cum_reward=([-\d.]+).*inserted=\[([\d.]+),([\d.]+)\]'

# Collect episodes with their max insertion depths
episodes_data = {}

for worker_log in sorted(worker_logs_dir.glob('worker_*.log')):
    worker_num = int(worker_log.stem.split('_')[1])
    
    with open(worker_log, 'r') as f:
        for line in f:
            match = re.search(step_pattern, line)
            if match:
                ep = int(match.group(1))
                ep_step = int(match.group(2))
                reward = float(match.group(3))
                cum_reward = float(match.group(4))
                insert_1 = float(match.group(5))
                insert_2 = float(match.group(6))
                
                # Track by (worker, episode)
                key = (worker_num, ep)
                if key not in episodes_data:
                    episodes_data[key] = {
                        'max_insert_1': 0,
                        'max_insert_2': 0,
                        'steps': 0,
                        'final_reward': 0
                    }
                
                episodes_data[key]['max_insert_1'] = max(episodes_data[key]['max_insert_1'], insert_1)
                episodes_data[key]['max_insert_2'] = max(episodes_data[key]['max_insert_2'], insert_2)
                episodes_data[key]['steps'] = ep_step
                episodes_data[key]['final_reward'] = cum_reward

# Convert to sorted list
episodes = []
for (worker, ep), data in episodes_data.items():
    episodes.append({
        'worker': worker,
        'episode': ep,
        'max_insertion': data['max_insert_1'],  # Using first device
        'steps': data['steps'],
        'total_reward': data['final_reward']
    })

episodes.sort(key=lambda x: (x['episode'], x['worker']))

print(f'Total episodes with insertion data: {len(episodes)}')
print()

# Divide into phases
total_eps = len(episodes)
early_cutoff = total_eps // 3
mid_cutoff = 2 * total_eps // 3

early_eps = episodes[:early_cutoff]
mid_eps = episodes[early_cutoff:mid_cutoff]
late_eps = episodes[mid_cutoff:]

print("=" * 80)
print("INSERTION DEPTH ANALYSIS BY TRAINING PHASE")
print("=" * 80)
print()

def analyze_phase(name, eps_list):
    insertions = [e['max_insertion'] for e in eps_list]
    rewards = [e['total_reward'] for e in eps_list]
    ep_range = (eps_list[0]['episode'], eps_list[-1]['episode'])
    
    print(f"{name} PHASE (episodes {ep_range[0]}-{ep_range[1]}, n={len(eps_list)}):")
    print(f"  Mean insertion:   {statistics.mean(insertions):8.2f} mm")
    print(f"  Median insertion: {statistics.median(insertions):8.2f} mm")
    print(f"  Std dev:          {statistics.stdev(insertions):8.2f} mm")
    print(f"  Min:              {min(insertions):8.2f} mm")
    print(f"  Max:              {max(insertions):8.2f} mm")
    print(f"  Avg reward:       {statistics.mean(rewards):8.4f}")
    print()

analyze_phase("EARLY", early_eps)
analyze_phase("MID", mid_eps)
analyze_phase("LATE", late_eps)

# Trend over time
print("=" * 80)
print("INSERTION TREND OVER TIME (10 segments)")
print("=" * 80)
print()

seg_size = len(episodes) // 10
print("Seg | Episode Range | Avg Insertion | Median | Max | Avg Reward")
print("-" * 75)

for i in range(10):
    start_idx = i * seg_size
    end_idx = len(episodes) - 1 if i == 9 else (i + 1) * seg_size - 1
    seg = episodes[start_idx:end_idx+1]
    
    insertions = [e['max_insertion'] for e in seg]
    rewards = [e['total_reward'] for e in seg]
    ep_start = seg[0]['episode']
    ep_end = seg[-1]['episode']
    
    print(f" {i+1:2} | {ep_start:3}-{ep_end:3}          | {statistics.mean(insertions):13.2f} | {statistics.median(insertions):6.2f} | {max(insertions):6.2f} | {statistics.mean(rewards):10.4f}")

# Overall change
print()
print("=" * 80)
print("OVERALL INSERTION TRAJECTORY")
print("=" * 80)

first_100 = episodes[:100] if len(episodes) >= 100 else episodes[:len(episodes)//2]
last_100 = episodes[-100:]

first_ins = statistics.mean(e['max_insertion'] for e in first_100)
last_ins = statistics.mean(e['max_insertion'] for e in last_100)
change_pct = (last_ins - first_ins) / first_ins * 100 if first_ins != 0 else 0

print(f"First 100 episodes avg insertion: {first_ins:.2f} mm")
print(f"Last 100 episodes avg insertion:  {last_ins:.2f} mm")
print(f"Change: {change_pct:+.1f}%")
print()

if last_ins < first_ins * 0.5:
    print("WARNING: Insertion depth dropped by >50% - POLICY COLLAPSE")
elif last_ins > first_ins * 1.2:
    print("GOOD: Policy improved insertion depth by >20%")
else:
    print("NEUTRAL: Insertion depth remained stable")
