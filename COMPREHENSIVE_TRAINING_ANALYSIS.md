# Comprehensive Training Analysis - ENV (Original)
## Run: 2026-02-03_003538_diag_debug_test4

---

## Executive Summary

**SEVERE POLICY COLLAPSE DETECTED**

The policy started with deep exploration (avg 743mm insertion during early training) but collapsed to shallow insertion behavior (avg 221mm in late training), representing a **70% performance degradation**. This confirms the policy learned a degenerate strategy.

---

## 1. EPISODE REWARD ANALYSIS

### By Training Phase:
| Phase | Episodes | Mean Reward | Median | Std Dev | Min | Max |
|-------|----------|-------------|--------|---------|-----|-----|
| **EARLY** | 1-4 (n=492) | -4.2795 | -4.9337 | 1.61 | -5.01 | +0.90 |
| **MID** | 4-9 (n=492) | -4.4670 | -4.9310 | 1.40 | -5.01 | +0.89 |
| **LATE** | 9-36 (n=493) | -4.4030 | -4.9004 | 1.43 | -5.01 | +0.85 |

### Overall Trajectory:
- **First 100 episodes**: -3.78 avg reward
- **Last 100 episodes**: -3.79 avg reward
- **Change**: -0.2% (essentially STABLE)

### Reward Trend Over Time (10 segments):
```
Seg  1:  -3.91 avg reward
Seg  2:  -4.41
Seg  3:  -4.44
Seg  4:  -4.40
Seg  5:  -4.39
Seg  6:  -4.52
Seg  7:  -4.53
Seg  8:  -4.59
Seg  9:  -4.71
Seg 10:  -3.95
```

**Key Finding**: Episode rewards remained **remarkably stable** throughout training, hovering around -4.4. This masked the underlying policy collapse because shallow insertions still accumulate similar total penalties.

---

## 2. INSERTION DEPTH ANALYSIS

### By Training Phase:
| Phase | Episodes | Mean Depth | Median | Std Dev | Min | Max |
|-------|----------|------------|--------|---------|-----|-----|
| **EARLY** | 1-9 (n=140) | **682.82 mm** | 898.36 | 282.20 | 114.66 | 900.00 |
| **MID** | 9-18 (n=140) | **312.63 mm** | 162.67 | 281.82 | 27.28 | 899.99 |
| **LATE** | 18-37 (n=140) | **202.03 mm** | 160.09 | 187.66 | 18.78 | 899.96 |

### Overall Trajectory:
- **First 100 episodes**: 743.22 mm avg insertion
- **Last 100 episodes**: 221.37 mm avg insertion
- **Change**: **-70.2% COLLAPSE**

### Insertion Trend Over Time (10 segments):
```
Seg  1:  811.18 mm (median 899.60)
Seg  2:  707.30 mm (median 898.12)
Seg  3:  619.68 mm (median 714.59)
Seg  4:  381.52 mm (median 214.17) ← Sharp drop
Seg  5:  356.72 mm (median 196.88)
Seg  6:  288.06 mm (median 157.89)
Seg  7:  199.87 mm (median 157.14)
Seg  8:  159.44 mm (median 154.49) ← Stabilized low
Seg  9:  171.45 mm (median 163.20)
Seg 10:  296.38 mm (median 178.45)
```

**Key Finding**: The policy learned to insert to ~160-220mm and then either freeze or oscillate. This represents a **catastrophic failure** to reach targets (typically at 400-600mm depth).

---

## 3. LOSS & TRAINING DYNAMICS

### Q-Loss and Policy Loss (10 segments):
| Segment | Steps (K) | Q1 Loss | Q2 Loss | Policy Loss | Alpha | Q1 Mean | Entropy |
|---------|-----------|---------|---------|-------------|-------|---------|---------|
| 1 | 95-164 | 0.0606 | 0.0594 | -18.71 | **0.6101** | 17.26 | +2.46 |
| 2 | 164-246 | 0.0347 | 0.0353 | -33.10 | 0.2060 | **32.62** | +2.56 |
| 3 | 246-347 | 0.0235 | 0.0247 | -32.52 | 0.0696 | 32.38 | +2.61 |
| 4 | 347-444 | 0.0178 | 0.1197 | -27.27 | 0.0237 | 27.25 | +2.41 |
| 5 | 444-567 | 0.0068 | 0.0057 | -21.85 | 0.0080 | 21.86 | +2.47 |
| 6 | 567-667 | 0.0038 | 0.0039 | -17.23 | 0.0027 | 17.25 | +2.50 |
| 7 | 667-763 | 0.0020 | 0.0022 | -13.43 | 0.0009 | 13.45 | +1.84 |
| 8 | 763-861 | 0.0011 | 0.0013 | -10.41 | 0.0003 | 10.43 | +0.07 |
| 9 | 861-905 | 0.0010 | 0.0011 | -7.81 | 0.0001 | 7.82 | **-1.83** |
| 10 | 905-987 | 0.0012 | 0.0012 | -5.78 | **0.0001** | **5.79** | **-4.67** |

### Key Metrics Evolution:
| Metric | Start | Peak/End | Change |
|--------|-------|----------|--------|
| **Q1 Mean** | 17.26 | 32.62 → 5.79 | Peak +89%, then **-82% collapse** |
| **Alpha** | 0.61 | 0.00013 | **-99.98% (exploration died)** |
| **Entropy** | +2.46 | -4.67 | **Complete determinism** |
| **Q1 Grad Norm** | 17.12 | 0.75 | -95.6% (learning stalled) |
| **Policy Grad Norm** | 26.13 | 5.17 | -80.2% |

---

## 4. PROBE STATE ANALYSIS (Fixed Start States)

Evolution of Q-values and actions for the **same fixed states** over training:

| Update Step | Avg Q1 | Avg Q2 | Trans Mean | Trans Std |
|-------------|--------|--------|------------|-----------|
| 100 | 1.62 | 1.69 | +0.0009 | 0.886 |
| 5,100 | **31.42** | 31.42 | +0.0006 | 0.876 |
| 10,100 | **35.39** | **35.42** | -0.0024 | 0.875 |
| 20,100 | 24.97 | 24.64 | -0.2338 | 0.857 |
| 35,100 | 11.63 | 11.62 | +0.3474 | 0.884 |
| 45,100 | 6.88 | 6.86 | **-2.45** | 0.561 |
| 49,200 | 5.49 | 5.47 | **-2.66** | **0.654** |

**Critical Finding**: For the SAME start state:
- Q-value estimate: 35.4 → 5.5 (**-84% collapse**)
- Translation action: 0.0 → **-2.66** (learned to RETRACT)
- Action std: 0.886 → 0.654 (less exploration)

This proves the critics are accurately learning the degenerate policy's low value, confirming the collapse is real, not a critic estimation error.

---

## 5. ROOT CAUSE ANALYSIS

### The Degenerate Strategy

The policy discovered a **local optimum**:

1. **Early Training (Episodes 1-3)**:
   - Random exploration → avg 811mm insertion
   - Q-values learn: "Deep insertion gets some positive rewards but many step penalties"
   - Q-values: ~17-35

2. **Mid Training (Episodes 4-11)**:
   - Policy starts optimizing
   - Discovers: "Insert ~200-400mm, then stop/oscillate"
   - This minimizes step penalties while occasionally hitting early waypoints
   - Q-values peak at ~32, then start declining

3. **Late Training (Episodes 12+)**:
   - Alpha collapses to near-zero (0.0001)
   - Policy locks into: "Insert ~160-220mm, then freeze/oscillate"
   - Q-values accurately reflect this shallow strategy: ~5.8
   - Entropy becomes negative (-4.67) = extreme determinism

### Why This Happened

**Reward Structure Flaw**:
```
Total Reward = Waypoint Progress + Step Penalty + Target Bonus

For shallow insertion (~200mm):
  - Waypoint rewards: ~0.3 to 0.6 (few waypoints crossed)
  - Step penalty: -0.001 × ~500 steps = -0.5
  - Target bonus: 0 (never reached)
  - NET: ~-0.2 to -0.4

For deep insertion (~700mm):
  - Waypoint rewards: ~2.0 to 3.0 (many waypoints crossed)
  - Step penalty: -0.001 × ~900 steps = -0.9
  - Target bonus: +5.0 (if reached)
  - NET: ~+4.0 to +6.0 (if successful), or ~+1.0 to +2.0 (if failed)
  
BUT: Deep insertion is HARD (branching, navigation)
     Shallow insertion is EASY (straight line, minimal risk)
```

The policy chose the **safe local optimum** over the **risky global optimum**.

---

## 6. SAC ALGORITHM BEHAVIOR

### Entropy Temperature (Alpha) Collapse

SAC's auto-tuning of alpha is supposed to maintain exploration, but:

- Alpha: 0.61 → 0.00013 (factor of 4,615x decrease)
- Target entropy: Not visible in logs, likely around -2 to -4
- Alpha loss became positive (+6.02 in late training), trying to INCREASE alpha
- But alpha is so small it barely responds

This indicates SAC's entropy regularization **failed** to prevent the collapse.

### Gradient Norms

- Q-network gradients: 17 → 0.75 (learning slowed drastically)
- Policy gradients: 26 → 5.17 (updates became small)

The networks converged to a stable but degenerate solution.

---

## 7. CONCLUSION

**Policy Status**: COLLAPSED to degenerate shallow-insertion strategy

**Evidence**:
1. ✅ Insertion depth: 743mm → 221mm (-70%)
2. ✅ Q-values: 17-35 → 5.8 (-82% from peak)
3. ✅ Alpha: 0.61 → 0.0001 (-99.98%)
4. ✅ Entropy: +2.46 → -4.67 (deterministic)
5. ✅ Probe state actions: 0.0 → -2.66 (retraction)
6. ✅ Episode rewards: Stable but LOW (-3.8), masking the collapse

**The policy learned**: "Insert ~200mm and stop" instead of "Navigate to target at ~500mm"

**Why rewards didn't reflect this**: The step penalty (-0.001) is so small that 500 steps vs 900 steps only differs by -0.4 in total penalty. The stable episode rewards (-3.8) disguised the fact that the policy was doing far less work.

**Next Steps**:
1. Increase step penalty OR decrease waypoint spacing to make shallow insertion less attractive
2. Add minimum depth requirement before episodic reward
3. Increase target entropy to maintain exploration longer
4. Consider curriculum learning (start with easy targets, gradually increase depth)
