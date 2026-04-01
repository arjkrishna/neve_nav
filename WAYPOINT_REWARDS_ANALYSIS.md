# Waypoint Rewards Analysis: ENV2 vs ENV3

## Summary of Parameters

| Environment | Waypoint Spacing | Branch Increment | Step Penalty |
|-------------|------------------|------------------|--------------|
| **ENV2** | 10mm | 0.1 | -0.001 |
| **ENV3** | 5mm | 1.0 | -0.001 |

## Path Assumptions
- **Total path length**: ~400mm from insertion to target
- **Branch structure**: 2 segments
  - Segment 1 (Trunk): 0-200mm
  - Segment 2 (Target Branch): 200-400mm

---

## ENV2 Waypoint Rewards (10mm spacing, 0.1 increment)

### Branch 0 (Trunk: 0-200mm, 20 waypoints)

| Distance | Waypoint # | Score | Cumulative Steps | Step Penalty | **Net Reward** |
|----------|------------|-------|------------------|--------------|----------------|
| 0mm | 1 | 0.0000 | 0 | 0.000 | **0.0000** |
| 10mm | 2 | 0.0047 | 4 | -0.004 | **0.0007** |
| 20mm | 3 | 0.0095 | 8 | -0.008 | **0.0015** |
| 30mm | 4 | 0.0142 | 12 | -0.012 | **0.0022** |
| 40mm | 5 | 0.0189 | 16 | -0.016 | **0.0029** |
| 50mm | 6 | 0.0237 | 20 | -0.020 | **0.0037** |
| 100mm | 11 | 0.0474 | 40 | -0.040 | **0.0074** |
| 150mm | 16 | 0.0711 | 60 | -0.060 | **0.0111** |
| 190mm | 20 | 0.0900 | 76 | -0.076 | **0.0140** |

### Branch 1 (Target Branch: 200-400mm, 20 waypoints)

| Distance | Waypoint # | Score | Cumulative Steps | Step Penalty | **Net Reward** |
|----------|------------|-------|------------------|--------------|----------------|
| 200mm | 21 | **0.1000** *(+0.01 jump)* | 80 | -0.080 | **0.0200** |
| 250mm | 26 | 0.1237 | 100 | -0.100 | **0.0237** |
| 300mm | 31 | 0.1474 | 120 | -0.120 | **0.0274** |
| 350mm | 36 | 0.1711 | 140 | -0.140 | **0.0311** |
| 390mm | 40 | 0.1900 | 156 | -0.156 | **0.0340** |

### ENV2 Final Result at 400mm:
- **Waypoint Score**: 0.19
- **Step Penalty**: -0.156
- **Net Reward**: **+0.034**
- **WITH Target Bonus (+1.0)**: +1.034

### Why ENV2 Failed (stuck at ~47mm):
- At 47mm: Net reward = +0.0029 (barely positive)
- Rewards grow VERY slowly (0.0047 per waypoint)
- Step penalty accumulates faster than waypoint rewards early on
- Policy learned to stop around 40-50mm where reward is neutral
- **The reward gradient is too weak to encourage deep exploration**

---

## ENV3 Waypoint Rewards (5mm spacing, 1.0 increment)

### Branch 0 (Trunk: 0-200mm, 40 waypoints)

| Distance | Waypoint # | Score | Cumulative Steps | Step Penalty | **Net Reward** |
|----------|------------|-------|------------------|--------------|----------------|
| 0mm | 1 | 0.0000 | 0 | 0.000 | **0.0000** |
| 5mm | 2 | 0.0254 | 2 | -0.002 | **0.0234** |
| 10mm | 3 | 0.0508 | 4 | -0.004 | **0.0468** |
| 15mm | 4 | 0.0762 | 6 | -0.006 | **0.0702** |
| 20mm | 5 | 0.1015 | 8 | -0.008 | **0.0935** |
| 25mm | 6 | 0.1269 | 10 | -0.010 | **0.1169** |
| 30mm | 7 | 0.1523 | 12 | -0.012 | **0.1403** ⭐ |
| 35mm | 8 | 0.1777 | 14 | -0.014 | **0.1637** |
| 40mm | 9 | 0.2031 | 16 | -0.016 | **0.1871** |
| 50mm | 11 | 0.2538 | 20 | -0.020 | **0.2338** |
| 100mm | 21 | 0.5077 | 40 | -0.040 | **0.4677** |
| 150mm | 31 | 0.7615 | 60 | -0.060 | **0.7015** |
| 195mm | 40 | 0.9900 | 78 | -0.078 | **0.9120** |

### Branch 1 (Target Branch: 200-400mm, 40 waypoints)

| Distance | Waypoint # | Score | Cumulative Steps | Step Penalty | **Net Reward** |
|----------|------------|-------|------------------|--------------|----------------|
| 200mm | 41 | **1.0000** *(+0.01 jump)* | 80 | -0.080 | **0.9200** |
| 250mm | 51 | 1.2538 | 100 | -0.100 | **1.1538** |
| 300mm | 61 | 1.5077 | 120 | -0.120 | **1.3877** |
| 350mm | 71 | 1.7615 | 140 | -0.140 | **1.6215** |
| 395mm | 80 | 1.9900 | 158 | -0.158 | **1.8320** |

### ENV3 Final Result at 400mm:
- **Waypoint Score**: 1.99
- **Step Penalty**: -0.158
- **Net Reward**: **+1.832**
- **WITH Target Bonus (+1.0)**: +2.832

### Why ENV3 Failed (stuck at 7-30mm):
- At 30mm: Net reward = **+0.14** (STRONG positive signal!)
- By 30mm, agent has already achieved 14% of max waypoint reward
- This created a **local optimum trap**:
  - **"Safe" strategy**: Insert to 30mm, oscillate, maintain +0.14 reward
  - **"Risky" strategy**: Continue to 400mm, risk wrong branches, uncertain outcome
- Policy converged to the safe local optimum
- **The early waypoints were TOO rewarding relative to the effort required**

---

## The Fundamental Problem

### ENV2: Rewards Too Small
```
Waypoint reward at 400mm: +0.19
Step penalty at 400mm:    -0.16
Net reward:                +0.03 (barely positive)
```
**Result**: Policy stops early (~47mm) where reward is neutral.

### ENV3: Early Waypoints Too Rewarding
```
Reward at 30mm:   +0.14 (safe, certain)
Reward at 400mm:  +1.83 (risky, uncertain)
Risk-adjusted:    30mm looks better!
```
**Result**: Policy exploits local optimum at 7-30mm instead of exploring deeper.

---

## What We Learned

1. **Heatup exploration easily reaches 1300mm** - the environment allows deep insertion
2. **All three trained policies INSERT LESS after training** - they learned conservative strategies
3. **Reward shaping is anti-correlated with the goal**:
   - Step penalties discourage depth
   - Wrong branch penalties discourage exploration
   - Early termination minimizes accumulated negative reward
   - Local optima trap policies in shallow insertion

4. **The reward structure needs fundamental redesign**:
   - Rewards should increase exponentially with depth (not linearly)
   - Step penalty should be offset by depth bonus
   - Wrong branch penalties should be recoverable
   - No exploitable local optima should exist before the target

---

## Comparison to Target Requirement

**Target depth**: 400-500mm

| Environment | Heatup Avg | Trained Avg | Performance |
|-------------|-----------|-------------|-------------|
| ENV | 1311mm | 60mm | -95% (collapse) |
| ENV2 | 1307mm | 47mm | -96% (collapse) |
| ENV3 | 1306mm | 7-30mm | -98% (worse collapse) |

**All policies failed to learn sustained forward insertion.**
