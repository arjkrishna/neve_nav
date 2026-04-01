# diagnose_collapse.py Fix Summary

## Issue Identified by User

**User observation**: "Is that a critic spike; seems like it started with a value and then gradually decreased"

**User was correct**: The tool was flagging normal neural network training startup (high initial losses that decrease rapidly) as a "critic spike" pathology.

---

## Root Cause

### Baseline Calculation Flaw

**OLD behavior**:
```python
# Baseline from first 200 updates
base_n = max(20, min(200, n // 10))
base_q1l = _safe_median(q1l[:base_n])  # Includes startup losses
```

- Computed median from first 200 updates
- These updates include **normal training startup** where losses go from ~6.0 → 0.06
- Resulted in baseline ~0.06 (the stabilized value)
- Comparing update=1 (loss=3.77) to this baseline flagged it as "63× baseline spike"

**Problem**: This is comparing training **start** to trained **state** - not a spike, just normal learning.

---

## The Fix

### 1. Skip Startup Phase for Baseline

```python
# NEW: Skip first 100 updates (startup), use 100-1100 for baseline
startup_skip = min(100, n // 10)
base_start = startup_skip
base_end = min(base_start + 1000, n)

base_q1l = _safe_median(q1l[base_start:base_end])  # Stable training only
```

**Why 100 updates?**
- Losses decrease from ~6.0 to ~0.06 in first 50-100 updates
- This is normal neural network initialization behavior
- Baseline should be from **stable** training, not startup

### 2. Skip Startup for Spike Detection

```python
# Only check for spikes AFTER startup phase
for i, (ql1, ql2) in enumerate(zip(q1l, q2l)):
    if i < startup_skip:
        q_spike_pred.append(False)  # Don't flag startup
    else:
        q_spike_pred.append((ql1 > spike_factor * base_q1l) or ...)
```

**Why?**
- A "spike" means: normal → abnormal jump → return/persist
- Startup is **expected high → low**, not a spike
- We only want to detect **post-stabilization** instability

### 3. Document Baseline Calculation

Added to summary output:
```python
baseline=dict(
    ...,
    computed_from_updates=f"{base_start}-{base_end}",
    startup_skip=startup_skip
)
```

Users can now see what was considered "baseline".

---

## Verification Results

### Test4

**BEFORE FIX**:
```
Classification: critic_instability_precedes_policy
Earliest event: critic_loss_spike at update=1, explore=95000
```

**AFTER FIX**:
```
Classification: policy_collapse_or_suboptimal_attractor
Earliest event: actor_actions_negative_on_batch_states at update=4500, explore=158000
```

### Test5

**BEFORE FIX**:
```
Classification: critic_instability_precedes_policy
Earliest event: critic_loss_spike at update=1, explore=95000
```

**AFTER FIX**:
```
Classification: policy_collapse_or_suboptimal_attractor
Earliest event: actor_actions_negative_on_batch_states at update=7200, explore=199000
```

---

## Corrected Event Timeline

| Event | test4 | test5 | Notes |
|-------|-------|-------|-------|
| **Actor actions negative** | update=4,500 | update=7,200 | **← EARLIEST (root cause)** |
| Replay actions negative | update=14,000 | update=10,100 | Policy spreading to buffer |
| Critic loss spike | update=16,239 | update=17,700 | **After** policy problems |
| Entropy drop | update=16,307 | update=17,700 | Policy becoming deterministic |
| Alpha shutdown | update=20,947 | update=20,947 | Consequence of collapse |
| Policy grad collapse | update=56,893 | update=70,000 | Full convergence |

---

## Key Insights

### 1. Original Manual Analysis Was More Accurate

**Comprehensive manual analysis** (`COMPREHENSIVE_TRAINING_ANALYSIS.md`) correctly identified:
- Root cause: **reward shaping allows local optimum**
- Mechanism: **policy collapse**, not critic instability
- Timeline: Policy discovers shallow insertion is rewarded around 4-7k updates

### 2. Automated Tool Had Domain-Unaware Baseline

The tool's initial baseline calculation didn't account for **universal neural network training behavior**:
- All networks start with high loss (random initialization)
- Losses decrease rapidly in first 50-100 updates
- Baseline should be from **stable training**, not startup

### 3. User Expertise Caught False Positive

User immediately recognized that losses "started with a value and then gradually decreased" = **normal training**, not a spike.

This highlights the importance of **domain knowledge validation** of automated diagnostics.

---

## Implications for Future Diagnosis

### What the Tool NOW Does Correctly

✅ **Skips startup phase** (first 100 updates) for baseline calculation  
✅ **Computes baseline from stable training** (updates 100-1100)  
✅ **Only flags post-stabilization spikes** as pathological  
✅ **Correctly classifies both test4 and test5** as policy collapse  
✅ **Aligns with original manual analysis**  

### Recommended Next Steps

1. **Reward shaping investigation**:
   - Why does shallow insertion (7-30mm) yield positive rewards?
   - How to make deeper insertion more rewarding?
   - Should step penalty be adjusted?

2. **Alpha tuning**:
   - Alpha collapses to ~0.0001 by update 21k
   - Consider alpha lower bound or slower annealing

3. **Exploration incentives**:
   - Policy exploits local optimum without exploring deeper
   - Consider curiosity bonus or depth-based rewards

---

## Lesson Learned

**Automated diagnostics must incorporate domain knowledge about normal behavior patterns.**

In this case:
- Neural networks ALWAYS start with high losses
- This is a feature, not a bug
- "Spike" detection must account for expected startup behavior

**The user's intuition was correct**: A gradual decrease from high to low is normal training, not a spike.

---

## Files Modified

- `diagnose_collapse.py`:
  - Lines 525-548: Baseline calculation (skip startup)
  - Lines 549-553: Entropy baseline std calculation (use new range)
  - Lines 585-600: Critic spike detection (skip startup phase)
  - Lines 674-682: Summary includes baseline calculation metadata

---

## Validation

Both test4 and test5 now produce:
- ✅ Correct classification: `policy_collapse_or_suboptimal_attractor`
- ✅ Correct earliest event: Actor actions negative (4.5k-7.2k updates)
- ✅ Correct timeline: Policy problems → replay problems → alpha collapse
- ✅ Matches comprehensive manual analysis
- ✅ No false positive on normal training startup

**The tool is now ready for production use on future training runs.**
