# Causal Uncertainty Model (Lee et al., 2015)

Python implementation of the Bayesian causal uncertainty agent for the
one‑shot vs incremental learning paradigm in:

> Lee SW, O'Doherty JP, Shimojo S (2015)  
> Neural Computations Mediating One-Shot Learning in the Human Brain.  
> *PLoS Biology*, 13(4): e1002137.

This folder focuses on the *computational model* and *task generator* used
to simulate the behavioral paradigm.

## Overview

The causal uncertainty model implements a Bayesian agent that learns about causal relationships between stimuli and outcomes. The key innovation is that the agent uses its own uncertainty about causal strength to adaptively adjust learning rates—when uncertainty is high, the agent learns more from single observations (one-shot learning).

### Model Components

1. **Dirichlet Belief State**: The agent maintains Dirichlet concentration parameters (α) for each stimulus type, representing beliefs about causal strength.

2. **Causal Uncertainty**: Computed as the variance of the Dirichlet distribution for each stimulus. Higher uncertainty means the agent is less confident about the causal strength.

3. **Adaptive One-Shot Learning Rate**: The learning rate for each stimulus is determined by a softmax over causal uncertainty values, scaled by a temperature parameter. This means stimuli with higher uncertainty get higher learning rates.

4. **Update Rules**: The model supports two update modes:
   - **"sole"**: Presence/absence effect—stimulus appears in block (1) or not (0)
   - **"additive"**: Count effect—number of times stimulus appears in block

5. **Primacy and Recency Effects**: Additional weights for stimuli appearing at the beginning or end of each block.

## Files

- `task_lee2015.py`  
  Schedule generator for the one‑shot / incremental task. Produces a
  `pandas.DataFrame` describing which stimuli and outcomes occur in each
  block and round (used as input to the model).

- `model_lee2015.py`  
  Implementation of the causal uncertainty agent (`CausalUncertaintyAgentLee2015`).
  The agent:
  - Maintains Dirichlet beliefs over causal strength for each stimulus.
  - Converts Dirichlet variance (“causal uncertainty”) into an adaptive
    one‑shot learning rate via a softmax.
  - Incorporates primacy and recency effects at the block level.
  - Can be fit to subject rating data via bounded optimization.

- `example_lee2015.py`  
  Comprehensive example demonstrating:
  1. Schedule generation (oneshot and incremental conditions)
  2. Simulating a "true" agent with known parameters
  3. Generating dummy subject ratings with noise
  4. Fitting model parameters to subject data (comparing sole vs additive modes)
  5. Using fitted model to make predictions on new experiments
  6. Plotting block-wise trajectories of ratings and learning rates

## Installation

This code uses only standard scientific Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib` (only needed for the trajectory plotting example)

You can install them (for example, via `pip`):

```bash
pip install numpy pandas scipy matplotlib
```

## Basic Usage

Run the examples from inside the `causal uncertainty model` directory.

- **Simulate rating and parameter fitting**

```bash
python example_lee2015.py
```

This will:

1. Generate a mixed schedule (20 oneshot + 20 incremental rounds)
2. Simulate a "true" agent with additive mode and known parameters
3. Generate noisy subject ratings from the true agent's trajectory
4. Fit both sole and additive modes to the subject data
5. Compare which mode fits better
6. Use the fitted model to make predictions on a new experiment
7. Plot block-wise trajectories showing how ratings and learning rates evolve

## Using the Components in Your Own Code

### 1. Generate a schedule

Use `OneShotIncrementalTaskLee2015` to build an experimental schedule:

```python
from task_lee2015 import OneShotIncrementalTaskLee2015

task = OneShotIncrementalTaskLee2015(seed=42)
conditions = ["oneshot", "incremental", "oneshot", "incremental"]
schedule_df = task.generate_experiment(conditions)
```

### 2. Run the model

Feed the schedule into `CausalUncertaintyAgentLee2015`:

```python
from model_lee2015 import CausalUncertaintyAgentLee2015

agent = CausalUncertaintyAgentLee2015()
alpha_hist, lr_hist = agent.run_experiment(schedule_df)
ratings = agent.current_ratings()
```

### 3. Fit to subject ratings

The model supports trial-wise (block-wise) fitting. Your subject data should
contain ratings for each stimulus at each block:

```python
import pandas as pd

# Subject data: one row per (round_idx, block_idx, stimulus)
subject_df = pd.DataFrame({
    "round_idx": [0, 0, 0, 0, 0, 0],
    "block_idx": [0, 0, 0, 1, 1, 1],
    "stimulus": [0, 1, 2, 0, 1, 2],
    "subject_rating": [0.3, 0.5, 0.2, 0.4, 0.4, 0.2]
})

# Fit a single mode
fit_result = agent.fit_to_subject_trialwise(
    schedule_df=schedule_df,
    subject_df=subject_df,
    cues_per_block=5,
    update_mode="additive",  # or "sole"
    maxiter=200
)
print("Best-fit params (lr, primacy, recency, temperature):", fit_result["params"])

# Or fit both modes and compare
fit_both = agent.fit_to_subject_trialwise_both(
    schedule_df=schedule_df,
    subject_df=subject_df,
    cues_per_block=5,
    maxiter=200
)
print("Best mode:", fit_both["best_mode"])
print("Sole loss:", fit_both["sole"]["loss"])
print("Additive loss:", fit_both["additive"]["loss"])
```

### 4. Extract trajectories

After running an experiment, you can extract the full block-wise trajectory:

```python
# Run experiment
agent.run_experiment(schedule_df, cues_per_block=5)

# Get trajectory DataFrame
traj_df = agent.get_trajectory_df(schedule_df)
# Contains: step, round_idx, block_idx, condition, stimulus, 
#           alpha, rating, os_lr

# Get current ratings
ratings = agent.current_ratings()  # Normalized causal ratings
```

### 5. Model Parameters

Key parameters when initializing `CausalUncertaintyAgentLee2015`:

- `lr` (float): Base learning rate scaling Dirichlet α updates (default: 0.1)
- `primacy` (float): Extra weight for first stimulus in block (default: 0.36)
- `recency` (float): Extra weight for last stimulus in block (default: 0.37)
- `temperature` (float): Softmax temperature for OS learning rate (default: 255.0)
- `alpha0_init` (float): Initial Dirichlet α for each stimulus (default: 1.0)
- `update_mode` (str): "sole" or "additive" (default: "sole")
- `reward_high` (float): Reward value when outcome == 1 (default: 1.0)
- `reward_low` (float): Reward value when outcome == 0 (default: -1.0)

## Task Structure

The task generator creates schedules with the following structure:

- **Stimuli**: 3 discrete cue types (encoded as 0, 1, 2)
- **Outcomes**: 2 discrete outcomes (encoded as 0, 1)
- **Round**: 5 blocks, each with 5 cue presentations + 1 outcome
- **Frequency constraints per round**:
  - Stimulus 0: 16 appearances
  - Stimulus 1: 8 appearances
  - Stimulus 2: 1 appearance (rare)
  - Outcome 0: 4 appearances
  - Outcome 1: 1 appearance (rare)

- **Conditions**:
  - **"oneshot"**: Rare stimulus and rare outcome occur in the same block
  - **"incremental"**: Rare stimulus and rare outcome occur in different blocks

The key experimental manipulation is whether the rare co-occurrence happens
in one block (oneshot) or is separated across blocks (incremental).

## Mathematical Details

### Causal Uncertainty

For each stimulus s, causal uncertainty is computed as:

```
CU_s = α_s (α_0 - α_s) / (α_0² (α_0 + 1))
```

where α_0 = Σ_s α_s is the sum of all Dirichlet concentration parameters.

### One-Shot Learning Rate

The OS learning rate for each stimulus is:

```
lr_s = softmax(τ × CU_s)
```

where τ is the temperature parameter. This ensures that stimuli with higher
uncertainty receive higher learning rates.

### Update Rule

For each block, the Dirichlet α values are updated:

```
Δα_s = lr × lr_s × reward × (x_s + primacy × I_primacy(s) + recency × I_recency(s))
```

where:
- `lr`: base learning rate
- `lr_s`: one-shot learning rate for stimulus s
- `reward`: +1 for outcome=1, -1 (or 0) for outcome=0
- `x_s`: presence (sole) or count (additive) of stimulus s in block
- `I_primacy(s)`: 1 if s is first in block, else 0
- `I_recency(s)`: 1 if s is last in block, else 0

## Citation

If you use this code in your work, please cite:

- Lee SW, O'Doherty JP, Shimojo S (2015) Neural Computations Mediating
  One-Shot Learning in the Human Brain. *PLoS Biology*, 13(4): e1002137.

## License

See the parent repository for license information.

## Contact

For support, extensions, or fitting guidance, please contact the original
author of this implementation (see the parent repository) or the authors of
the paper above.
