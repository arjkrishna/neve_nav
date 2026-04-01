# Complete Guide: Radius Information in stEVE Catheter Navigation

## 📋 Table of Contents
1. [Quick Summary](#quick-summary)
2. [What is Radius and Why It Matters](#what-is-radius-and-why-it-matters)
3. [How to Extract Radius](#how-to-extract-radius)
4. [Where Radius is Used in stEVE](#where-radius-is-used-in-steve)
5. [Agent vs Environment: Critical Distinction](#agent-vs-environment-critical-distinction)
6. [Impact on RL Training](#impact-on-rl-training)
7. [Practical Recommendations](#practical-recommendations)

---

## Quick Summary

### ⚡ The Essential Facts

**Q: Does stEVE use radius information?**
- ✅ **YES** - Critical for 6 different environment functions

**Q: Can the RL agent see radius?**
- ❌ **NO** - Agent only sees 2D fluoroscopy images

**Q: Why does radius matter if agent can't see it?**
- 💡 **Environment uses it** for rewards, target placement, and episode logic
- Wrong radius → Wrong environment feedback → Agent learns wrong policy

**Q: Can I use PTH files (which have no radius)?**
- ❌ **NO** - PTH only has coordinates, radius requires VMTK processing of VTP meshes

**Q: Best way to extract radius?**
- ⭐ **VMTK Python** for batch automation (50+ models)
- ⭐ **3D Slicer VMTK** for manual quality control (<10 models)

---

## What is Radius and Why It Matters

### The Two-Component System

stEVE uses **two separate representations** of vessel geometry:

```
╔═══════════════════════════════════════════════════╗
║              stEVE Vessel System                  ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  1. OBJ Mesh (Surface Geometry)                   ║
║     • Used by SOFA physics engine                 ║
║     • Defines vessel WALLS                        ║
║     • Collision detection & contact forces        ║
║                                                   ║
║  2. Centerlines + Radius (Logical Geometry)       ║
║     • Used by stEVE navigation logic              ║
║     • Defines vessel INTERIOR                     ║
║     • Branching, targets, rewards, truncation     ║
║                                                   ║
║  BOTH ARE REQUIRED!                               ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

### What is Radius?

**Radius = Maximum Inscribed Sphere Radius (MISR)**

At each centerline point, radius is the size of the largest sphere that:
- Fits inside the vessel
- Is centered on the centerline
- Is tangent to the vessel wall

```
Cross-section view:

    Vessel Wall
    ╱─────────╲
   │     ●     │  ← Centerline point
   │   ╱─╲   │  ← Maximum inscribed sphere
   │  │   │  │     (radius = 3.5mm)
   │   ╲─╱   │
    ╲─────────╱
    
    ← r=3.5mm →
```

### PTH Files vs VMTK Extraction

| Data Source | Has Coordinates? | Has Radius? | How Radius is Obtained |
|-------------|-----------------|-------------|------------------------|
| **PTH files** | ✅ YES | ❌ NO | N/A - just centerline coordinates |
| **VMTK extraction** | ✅ YES | ✅ YES | Computed from VTP surface mesh |

**Critical:** You cannot compute radius from coordinates alone - you need the surface mesh!

```
PTH file (XML):
<pos x="12.5" y="-8.3" z="145.2"/>  ← Just coordinates!
<pos x="12.7" y="-8.5" z="145.8"/>

VMTK extraction (JSON with radius):
{
  "controlPoints": [
    {"position": [12.5, -8.3, 145.2]},
    {"position": [12.7, -8.5, 145.8]}
  ],
  "measurements": [{
    "name": "Radius",
    "controlPointValues": [5.2, 5.1]  ← Computed from mesh!
  }]
}
```

---

## How to Extract Radius

### Option 1: VMTK Python (Recommended for Batch Processing)

**Pros:**
- ✅ Fully automated - no manual intervention
- ✅ Batch processing - process 50+ models overnight
- ✅ 100% reproducible - same script = same results
- ✅ Computes radius automatically (MISR)
- ✅ CI/CD friendly - runs headless on servers

**Cons:**
- ❌ Complex installation (conda recommended)
- ❌ No visual feedback - hard to verify quality
- ❌ Endpoint detection may fail on complex anatomy
- ❌ Harder to debug failures

**When to use:**
- Processing many (10+) models
- Need reproducibility
- Automated data pipelines
- Batch experiments

**Example:**
```bash
# Install VMTK
conda create -n vmtk python=3.9
conda activate vmtk
conda install -c vmtk vmtk

# Extract centerlines with radius
python vmr_processing_tools/extract_centerlines_vmtk.py vmr_downloads/0105_0001/

# Batch process all models
for model in vmr_downloads/*/; do
    python extract_centerlines_vmtk.py "$model"
done
```

---

### Option 2: 3D Slicer VMTK (Recommended for Quality)

**Pros:**
- ✅ Visual feedback - see results immediately
- ✅ Easy endpoint selection - click on mesh
- ✅ Interactive refinement - adjust and recompute
- ✅ Quality control built-in
- ✅ Easier installation (bundled in Slicer)
- ✅ Computes radius automatically (MISR)

**Cons:**
- ❌ Manual process - one model at a time
- ❌ Time consuming - 5-10 min per model
- ❌ Not scriptable (unless using Slicer CLI)
- ❌ Less reproducible - user choices vary

**When to use:**
- Processing few (<10) models
- Complex anatomies requiring careful endpoint selection
- Visual quality control is critical
- Publication-quality data
- Initial prototyping

**Workflow:**
1. Install 3D Slicer
2. View → Extension Manager → Search "SlicerVMTK" → Install
3. Load VTP mesh
4. Modules → VMTK → Extract Centerline
5. Mark source and target points (click on mesh)
6. Click Apply
7. Export as `.mrk.json`

---

### Option 3: PTH → JSON Conversion (NOT RECOMMENDED)

**Pros:**
- ✅ Instant - no computation needed
- ✅ Simple Python code
- ✅ Works offline

**Cons:**
- ❌ **NO RADIUS DATA** ← This is the killer!
- ❌ Cannot compute radius from coordinates alone
- ❌ Not suitable for realistic catheter navigation
- ❌ Breaks environment logic (see below)

**When to use:**
- ⚠️ Quick prototyping only (path planning without physics)
- ⚠️ NOT for training realistic navigation policies

**Do NOT use for:**
- ❌ DualDeviceNav-quality training
- ❌ Any RL training that uses PathLengthDelta reward
- ❌ Environments with VesselEnd truncation
- ❌ Evaluation with PathRatio metric

---

### Comparison Table

| Feature | VMTK Python | 3D Slicer VMTK | PTH Conversion |
|---------|-------------|----------------|----------------|
| **Automation** | ✅ Full | ⚠️ Manual | ✅ Simple |
| **Batch Processing** | ✅ Easy | ❌ Hard | ✅ Easy |
| **Visual Feedback** | ❌ None | ✅ Excellent | ❌ None |
| **Radius Included** | ✅ **YES** | ✅ **YES** | ❌ **NO** |
| **Training Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ Poor |
| **Time per Model** | ~2 min | ~8 min | ~10 sec |
| **Best For** | 10-50+ models | 1-10 models | NOT for training |

---

## Where Radius is Used in stEVE

Radius is used in **6 critical places** in the environment:

### 1. 🌳 Branching Detection (Setup/Reset)

**When:** Environment initialization and episode reset
**Function:** `calc_branching(branches, radii)`
**Purpose:** Build vessel topology graph

```python
# eve/eve/intervention/vesseltree/util/branch.py

def calc_branching(branches: List[Branch], radii: Union[float, List[float]]):
    """Detect where vessels connect by checking radius overlap"""
    for main_branch, main_radius in zip(branches, radii):
        for other_branch, other_radius in zip(branches, radii):
            # Check if branches overlap within radius
            points_in_main_branch = main_branch.in_branch(
                other_branch.coordinates, main_radius  # ← USES RADIUS!
            )
            if np.any(points_in_main_branch):
                # Create branching point where vessels connect
                branching_points.append(
                    BranchingPoint(coords, radius, [main_branch, other_branch])
                )
    return branching_points
```

**Why it matters:**
```
With radius:                  Without radius:
   Aorta                         Aorta
   │  r=10mm                     │  (unknown radius)
   ├──→ LCCA (r=3mm)             ???→ LCCA
   └──→ RCCA (r=3mm)             ???→ RCCA
   
✅ Branches connected          ❌ Can't detect connections
✅ Pathfinder works            ❌ Pathfinder fails
```

---

### 2. 🎯 Target Placement (Episode Reset)

**When:** Start of each episode
**Function:** `CenterlineRandom._in_excluded_branches()`
**Purpose:** Ensure targets are placed INSIDE navigable vessels

```python
# eve/eve/intervention/target/centerlinerandom.py

def _in_excluded_branches(self, coordinates, excluded_branches):
    """Check if target points are inside excluded branches"""
    in_branch = [False] * coordinates.shape[0]
    for branch_name in excluded_branches:
        branch = self.vessel_tree[branch_name]
        if isinstance(branch, BranchWithRadii):
            in_branch = branch.in_branch(coordinates)  # ← USES RADIUS!
    return in_branch

# BranchWithRadii.in_branch()
def in_branch(self, points):
    """Check if points are within vessel radius"""
    vectors = points - self.coordinates
    dist = np.linalg.norm(vectors, axis=-1)
    in_branch = np.any(dist < self.radii, axis=-1)  # ← USES RADII!
    return in_branch
```

**Why it matters:**
```
With radius:                  Without radius:
   Vessel r=5mm                  Centerline only
   ●─────────●                   ───────────
   │    ★   │  ← Target inside   ─────★──── ← Target in WALL!
   │  Reachable!                 ───────────  (Impossible!)
   ●─────────●                   

✅ Valid targets              ❌ Targets may be unreachable
✅ Agent can learn            ❌ Training fails
```

---

### 3. 🗺️ Pathfinding (Every Step)

**When:** Every environment step
**Function:** `BruteForceBFS.step()`
**Purpose:** Calculate optimal path from catheter tip to target

```python
# eve/eve/pathfinder/bruteforcebfs.py

def _init_vessel_tree(self):
    # Build navigation graph using branching points
    self._node_connections = self._initialize_node_connections(
        self.intervention.vessel_tree.branching_points  # ← USES BRANCHING!
    )
    
def step(self):
    # Find shortest path using branching graph
    path_length, path_points = self._get_shortest_path(
        position_branch, target_branch
    )
```

**Dependency chain:**
```
Pathfinder
  ↓
vessel_tree.branching_points
  ↓
calc_branching(branches, radii)
  ↓
REQUIRES RADIUS
```

**Why it matters:**
- Pathfinder provides "optimal path" for reward calculation
- If branching is wrong → pathfinding is wrong
- If pathfinding is wrong → rewards are wrong

---

### 4. 🎁 Reward Calculation (Every Step)

**When:** Every environment step
**Function:** `PathLengthDelta.step()`
**Purpose:** Guide agent toward optimal navigation

```python
# training_scripts/util/env.py (DualDeviceNav configuration)

# Primary reward component
path_delta = eve.reward.PathLengthDelta(pathfinder, factor=0.001)

# Reward equation
reward = PathLengthDelta + TargetReached + Step
       = -Δ(path_length) * 0.001 + 1.0 * reached + -0.005
```

```python
# eve/eve/reward/pathlengthdelta.py

def step(self):
    path_length = self.pathfinder.path_length  # ← Uses pathfinder
    delta = path_length - self._last_path_length
    self.reward = -delta * self.factor  # Reward for moving closer
    self._last_path_length = path_length
```

**Why it matters:**
- PathLengthDelta is the **primary guidance signal** for RL
- Computed **every single step** (20,000,000+ times in training)
- If pathfinder is wrong → rewards are wrong → agent learns wrong policy

**Training impact:**
```
DualDeviceNav training: 20,000,000 steps
PathLengthDelta computed: 20,000,000 times

With correct radius:
✅ Each reward guides agent correctly
✅ Agent learns optimal navigation policy

Without radius:
❌ 20,000,000 incorrect reward signals
❌ Agent learns from wrong feedback
❌ Training fails or converges to poor policy
```

---

### 5. 🛑 Truncation (Every Step, Training Only)

**When:** Every step during training
**Function:** `VesselEnd.truncated`
**Purpose:** End episode if catheter reaches vessel dead-end

```python
# eve/eve/truncation/vesselend.py

@property
def truncated(self):
    tip = self.intervention.fluoroscopy.tracking3d[0]
    return at_tree_end(tip, self.intervention.vessel_tree)

# eve/eve/intervention/vesseltree/vesseltree.py

def at_tree_end(point, vessel_tree):
    # Check if at branch endpoint
    if (min_idx == 0 or min_idx == branch_np.shape[0] - 1):
        # Check if endpoint connects to other branches
        end_is_open = True
        for branching_point in vessel_tree.branching_points:
            dist = np.linalg.norm(branching_point.coordinates - branch_point)
            if dist < branching_point.radius:  # ← USES RADIUS!
                end_is_open = False  # Connects to other branch
        return end_is_open
```

**Why it matters:**
- Prevents agent from pushing into walls
- Only active during training (not evaluation)
- Wrong radius → wrong termination → confusing training signal

---

### 6. 📊 Evaluation Metrics (Episode End)

**When:** End of each episode
**Function:** `PathRatio.step()`
**Purpose:** Measure navigation efficiency

```python
# eve/eve/info/pathratio.py

def step(self):
    trajectory_length = calculate_trajectory_length(...)  # Actual path
    optimal_path_length = self.pathfinder.path_length     # ← Uses pathfinder
    self.path_ratio = trajectory_length / optimal_path_length
```

**Why it matters:**
- Used to evaluate training quality
- Lower PathRatio = more efficient navigation
- If optimal_path is wrong → metric is meaningless

---

### Summary: Complete Usage

| Use Case | When | Function | Impact if Missing |
|----------|------|----------|-------------------|
| **Branching** | Setup/Reset | calc_branching() | ❌ Wrong topology |
| **Targets** | Episode reset | _in_excluded_branches() | ❌ Unreachable targets |
| **Pathfinding** | Every step | BruteForceBFS | ❌ Wrong optimal paths |
| **Rewards** | Every step | PathLengthDelta | ❌ Wrong guidance |
| **Truncation** | Every step (train) | at_tree_end() | ❌ Random termination |
| **Metrics** | Episode end | PathRatio | ❌ Meaningless evaluation |

**Result: 3 out of 5 reward components depend on radius!**

---

## Agent vs Environment: Critical Distinction

### What the Agent Sees (No Radius)

```python
# training_scripts/util/env.py

observation = eve.observation.ObsDict({
    "tracking": tracking,        # 2D fluoroscopy - catheter positions
    "target": target_state,      # 2D fluoroscopy - target position
    "last_action": last_action,  # Previous action
})
```

**Agent's inputs:**
- 📷 2D fluoroscopy images (catheter tracking points)
- 🎯 2D target position
- ⏮️ Last action

**Agent CANNOT see:**
- ❌ Vessel radius
- ❌ 3D geometry
- ❌ Centerlines
- ❌ Branching points
- ❌ Optimal path
- ❌ OBJ mesh

---

### What the Environment Uses (Radius Required)

```
┌──────────────────────────────────────────────────┐
│           ENVIRONMENT LOGIC                      │
│      (Hidden from agent, uses radius)            │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Branching detection    ← USES RADIUS        │
│  2. Target placement        ← USES RADIUS        │
│  3. Pathfinding             ← USES BRANCHING     │
│  4. Reward calculation      ← USES PATHFINDER    │
│  5. Truncation              ← USES RADIUS        │
│  6. Evaluation metrics      ← USES PATHFINDER    │
│                                                  │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│            AGENT INTERFACE                       │
│        (What agent sees - NO radius)             │
├──────────────────────────────────────────────────┤
│                                                  │
│  INPUT:  2D observations                         │
│  OUTPUT: Actions (translation, rotation)         │
│  FEEDBACK: Scalar reward                         │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

### Two Perspectives

**Agent's Perspective:**
```
Agent: "I see catheter on 2D screen at position X,Y"
Agent: "I see target at position X',Y'"
Agent: "If I push forward, I get reward +0.0021"
Agent: "I don't know WHY that's the reward"
Agent: "I just learn: this action → good outcome"
```

**Environment's Perspective:**
```
Env: "Catheter tip is at 3D position [65.2, -5.1, 35.8]"
Env: "This is 2.3mm from LCCA centerline (radius=3.5mm) → INSIDE vessel ✓"
Env: "Target placed using radius check → reachable ✓"
Env: "Optimal path = 127.5mm (using branching points) ✓"
Env: "Catheter moved 2.1mm closer → reward = +2.1 * 0.001 ✓"
Env: "Not at vessel end (checked with radius) ✓"
```

**Key insight:** Agent never sees the environment's calculations, but if those calculations are wrong (due to missing radius), the agent receives wrong feedback and learns incorrectly!

---

## Impact on RL Training

### DualDeviceNav Training Configuration

```python
HEATUP_STEPS = 500,000
TRAINING_STEPS = 20,000,000
TOTAL = 20,500,000 steps

# PathLengthDelta computed 20,500,000 times
# Each computation depends on pathfinder
# Pathfinder depends on branching points
# Branching points depend on radius
```

### Scenario A: With Proper Radius (Current)

```
Episode 1, Step 1:
  Env: Branching correct → Pathfinder correct → Optimal path = 127.5mm
  Env: Agent moved 2.1mm closer → Reward = +0.0021
  Agent: "Action A is good!"

Episode 1,000, Step 500:
  Agent: "I learned to navigate efficiently"
  Agent: "Turn left at bifurcation to reach LCCA"
  PathRatio: 1.15 (actual/optimal)

After 20M steps:
  ✅ Agent learned optimal navigation policy
  ✅ High success rate (>90%)
  ✅ Efficient paths (PathRatio ~1.1-1.2)
```

### Scenario B: Without Radius (Wrong)

```
Episode 1, Step 1:
  Env: Branching wrong → Pathfinder wrong → "Optimal path" = 250mm (WRONG!)
  Env: Agent moved 2.1mm but path changed by -5mm → Reward = +0.005 (WRONG!)
  Agent: "Action A is good!" (WRONG SIGNAL!)

OR:

Episode 1, Reset:
  Env: Target placed in wall (no radius check)
  Agent: "I can't reach this target no matter what I do"
  Agent: "Targets are random and impossible"

After 20M steps:
  ❌ Agent learned from 20M wrong signals
  ❌ Low success rate (<30%)
  ❌ Inefficient paths (PathRatio >2.0 or undefined)
  ❌ Training fails
```

---

### Impact Summary

| Training Aspect | With Radius | Without Radius |
|----------------|-------------|----------------|
| **Target Reachability** | ✅ Always reachable | ❌ May be in walls |
| **Reward Signals** | ✅ Correct guidance | ❌ Wrong guidance |
| **Episode Termination** | ✅ Logical boundaries | ❌ Random stops |
| **Pathfinding** | ✅ True optimal paths | ❌ Wrong or no paths |
| **Success Rate** | ⭐⭐⭐⭐⭐ >90% | ⭐ <30% |
| **PathRatio** | ⭐⭐⭐⭐⭐ 1.1-1.2 | ⭐ >2.0 or undefined |
| **Training Outcome** | ✅ Converges | ❌ Fails |

---

## Practical Recommendations

### For Your VMR Processing (51 Models)

**Goal:** Create DualDeviceNav-quality training data

**Recommended Approach:** Hybrid Strategy

```bash
# Step 1: Try VMTK Python automation on all models
for model in $(cat vmr_download_tools/vmr_filtered_models.txt); do
    echo "Processing $model..."
    python vmr_processing_tools/extract_centerlines_vmtk.py "vmr_downloads/$model/" \
        2>> vmtk_errors.log
done

# Step 2: Check which failed
grep "ERROR" vmtk_errors.log
# Expect ~10-20% failure rate on complex anatomies

# Step 3: Manually process failures in 3D Slicer
# - Open each failed model
# - Mark endpoints carefully (visual inspection)
# - Export JSON manually

# Step 4: Verify all have radius data
python verify_radius_data.py vmr_downloads/*/centerlines_vmtk/*.json
```

---

### Decision Tree

```
How many models?
├─ <5 models
│  └─ Use: 3D Slicer VMTK (manual)
│     Why: Best quality, visual control
│
├─ 5-20 models
│  └─ Use: Hybrid (try Python, fix failures with Slicer)
│     Why: Balance automation and quality
│
└─ >20 models
   └─ Use: VMTK Python (automated)
       Why: Batch processing, reproducibility
```

---

### Verification Script

```python
# verify_radius_data.py
import json
import sys
from pathlib import Path

def verify_radius(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    for markup in data.get("markups", []):
        for measure in markup.get("measurements", []):
            if measure["name"] == "Radius":
                radii = measure["controlPointValues"]
                print(f"✅ {json_file.name}")
                print(f"   Points: {len(radii)}")
                print(f"   Radius: {min(radii):.2f} - {max(radii):.2f} mm")
                return True
    
    print(f"❌ {json_file.name} - NO RADIUS DATA!")
    return False

# Check all JSON files
json_files = Path("vmr_downloads").rglob("*centerlines*.mrk.json")
results = [verify_radius(f) for f in json_files]

print(f"\n{sum(results)}/{len(results)} files have radius data")
if not all(results):
    sys.exit(1)
```

---

### Quality Checklist

Before using extracted centerlines for training:

- [ ] Radius data present in all JSON files
- [ ] Radius values reasonable (0.5mm - 15mm for neurovascular)
- [ ] Centerlines cover all target branches
- [ ] No gaps or disconnected segments
- [ ] Branching points detected correctly
- [ ] Visual inspection of at least 10% of models

---

## Final Summary

### Key Takeaways

1. **Radius is Critical**
   - Used in 6 places in environment logic
   - Required for correct RL training
   - Missing radius = broken training

2. **Agent Can't See Radius**
   - Agent only sees 2D fluoroscopy
   - Radius is environment ground truth
   - But wrong radius → wrong feedback → wrong learning

3. **PTH Files Are Insufficient**
   - PTH only has coordinates
   - Radius requires VMTK processing of VTP mesh
   - Cannot compute radius from coordinates alone

4. **Use VMTK for Extraction**
   - VMTK Python: Batch automation
   - 3D Slicer VMTK: Quality control
   - Both compute proper radius (MISR)

5. **Hybrid Approach is Best**
   - Start with Python automation
   - Manually fix failures with Slicer
   - Verify all files have radius

---

### Quick Reference

| Question | Answer |
|----------|--------|
| Does stEVE use radius? | ✅ YES - 6 places |
| Can agent see radius? | ❌ NO - only 2D images |
| Why radius matters? | Environment logic & rewards |
| Can I use PTH files? | ❌ NO - missing radius |
| Best extraction method? | VMTK Python or 3D Slicer |
| Training without radius? | ❌ FAILS - wrong feedback |

---

### Next Steps

```bash
# 1. Install VMTK
conda create -n vmtk python=3.9
conda activate vmtk
conda install -c vmtk vmtk

# 2. Extract centerlines with radius
python vmr_processing_tools/extract_centerlines_vmtk.py vmr_downloads/0105_0001/

# 3. Verify radius is present
python verify_radius_data.py

# 4. Process all models
for model in vmr_downloads/*/; do
    python extract_centerlines_vmtk.py "$model"
done

# 5. Use for training
python training_scripts/DualDeviceNav_train.py
```

**Don't compromise on data quality!** Radius is essential for realistic catheter navigation training. 🎯

