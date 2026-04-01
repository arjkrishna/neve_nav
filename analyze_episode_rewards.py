import re
import statistics
from pathlib import Path

run_dir = Path(r'training_scripts\results\eve_paper\neurovascular\full\mesh_ben\2026-02-03_003538_diag_debug_test4')
worker_logs_dir = run_dir / 'logs_subprocesses'

# Pattern to extract episode number and total reward
pattern = r'EPISODE_END \| ep=(\d+) \| steps=\d+ \| total_reward=([-\d.]+)'

# Collect all episodes from all workers
episodes = []

for worker_log in sorted(worker_logs_dir.glob('worker_*.log')):
    worker_num = int(worker_log.stem.split('_')[1])
    
    with open(worker_log, 'r') as f:
        content = f.read()
        matches = re.findall(pattern, content)
        
        for ep_num, reward in matches:
            episodes.append({
                'worker': worker_num,
                'episode': int(ep_num),
                'reward': float(reward)
            })

# Sort by episode number
episodes.sort(key=lambda x: (x['episode'], x['worker']))

print(f'Total episodes collected: {len(episodes)}')
print()

# Group into phases
total_eps = len(episodes)
early_cutoff = total_eps // 3
mid_cutoff = 2 * total_eps // 3

early_eps = episodes[:early_cutoff]
mid_eps = episodes[early_cutoff:mid_cutoff]
late_eps = episodes[mid_cutoff:]

print("=" * 80)
print("EPISODE REWARD ANALYSIS BY TRAINING PHASE")
print("=" * 80)
print()

# Calculate statistics for each phase
def analyze_phase(name, eps_list):
    rewards = [e['reward'] for e in eps_list]
    ep_range = (eps_list[0]['episode'], eps_list[-1]['episode'])
    
    print(f"{name} PHASE (episodes {ep_range[0]}-{ep_range[1]}, n={len(eps_list)}):")
    print(f"  Mean reward:   {statistics.mean(rewards):8.4f}")
    print(f"  Median reward: {statistics.median(rewards):8.4f}")
    print(f"  Std dev:       {statistics.stdev(rewards):8.4f}")
    print(f"  Min reward:    {min(rewards):8.4f}")
    print(f"  Max reward:    {max(rewards):8.4f}")
    print()

analyze_phase("EARLY", early_eps)
analyze_phase("MID", mid_eps)
analyze_phase("LATE", late_eps)

# Show trend over time in 10 segments
print("=" * 80)
print("REWARD TREND OVER TIME (10 segments)")
print("=" * 80)
print()

seg_size = len(episodes) // 10
print("Seg | Episode Range | Avg Reward | Median | Std Dev")
print("-" * 65)

for i in range(10):
    start_idx = i * seg_size
    end_idx = len(episodes) - 1 if i == 9 else (i + 1) * seg_size - 1
    seg = episodes[start_idx:end_idx+1]
    
    rewards = [e['reward'] for e in seg]
    ep_start = seg[0]['episode']
    ep_end = seg[-1]['episode']
    
    print(f" {i+1:2} | {ep_start:3}-{ep_end:3}          | {statistics.mean(rewards):10.4f} | {statistics.median(rewards):6.4f} | {statistics.stdev(rewards):7.4f}")

# Overall trajectory
print()
print("=" * 80)
print("OVERALL TRAJECTORY")
print("=" * 80)

first_100 = episodes[:100] if len(episodes) >= 100 else episodes[:len(episodes)//2]
last_100 = episodes[-100:]

first_avg = statistics.mean(e['reward'] for e in first_100)
last_avg = statistics.mean(e['reward'] for e in last_100)
change_pct = (last_avg - first_avg) / abs(first_avg) * 100 if first_avg != 0 else 0

print(f"First 100 episodes avg: {first_avg:.4f}")
print(f"Last 100 episodes avg:  {last_avg:.4f}")
print(f"Change: {change_pct:+.1f}%")
print()

if last_avg < first_avg * 0.5:
    print("WARNING: Reward dropped by >50% - SEVERE POLICY COLLAPSE")
elif last_avg < first_avg:
    print("WARNING: Reward decreased - POLICY DEGRADATION")
else:
    print("OK: Policy improved or maintained performance")
