# One-Shot Learning: Lee et al. (2015) Implementation

This repository contains implementations of the one-shot learning model and experimental paradigm from:

> Lee SW, O'Doherty JP, Shimojo S (2015)  
> Neural Computations Mediating One-Shot Learning in the Human Brain.  
> *PLoS Biology*, 13(4): e1002137.

The project includes computational models, task generators, and experimental simulation tools for studying one-shot vs. incremental causal learning.

## Project Overview

This repository is organized into two main components:

1. **Causal Inference Task** (MATLAB) - Experimental simulation scripts for running causal inference experiments with visual stimuli and participant responses
2. **Causal Uncertainty Model** (Python) - Bayesian causal uncertainty agent with Dirichlet priors that implements the computational model from the paper

## Repository Structure

```
OneShot-Lee2015-main/
├── Causal inference task/          # MATLAB experimental simulation
│   ├── causal_inference_exp.m      # Main experimental session script
│   ├── event_random.m              # Random event sequence generator
│   ├── given_event_made.m          # Progressive event generator
│   ├── Instruction.pdf             # Experimental instructions
│   ├── README.md                   # Detailed MATLAB component documentation
│   └── seed.vol1.egg, seed.vol2.egg # Random seed files (extract to ./seed/)
│
└── causal uncertainty model/       # Bayesian causal uncertainty agent
    ├── model_lee2015.py            # CausalUncertaintyAgentLee2015 class
    ├── task_lee2015.py             # Task schedule generator
    ├── example_lee2015.py          # Comprehensive usage example
    └── README.md                   # Detailed Python component documentation
```

## Installation

### Python Components

The Python components require standard scientific libraries:

```bash
pip install numpy pandas scipy matplotlib
```

### MATLAB Components

The MATLAB scripts require:
- **MATLAB** (R2014b or later recommended)
- **[Psychtoolbox-3](http://psychtoolbox.org/)** for visual stimulus presentation
- **Seed images**: Extract `seed.vol1.egg` and `seed.vol2.egg` to create `./seed/` directory

See the [Causal inference task README](Causal%20inference%20task/README.md) for detailed setup instructions.

## Quick Start

### 1. Causal Uncertainty Model (Python)

**Basic simulation and parameter fitting:**

```bash
cd "causal uncertainty model"
python example_lee2015.py
```

This will:
- Generate a mixed schedule (20 oneshot + 20 incremental rounds)
- Simulate a "true" agent with additive mode and known parameters
- Generate noisy subject ratings from the true agent's trajectory
- Fit both sole and additive modes to the subject data
- Compare which mode fits better
- Use the fitted model to make predictions on a new experiment
- Plot block-wise trajectories showing how ratings and learning rates evolve

**Using the model in your own code:**

```python
from task_lee2015 import OneShotIncrementalTaskLee2015
from model_lee2015 import CausalUncertaintyAgentLee2015

# Generate task schedule
task = OneShotIncrementalTaskLee2015(seed=42)
conditions = ["oneshot", "incremental", "oneshot", "incremental"]
schedule_df = task.generate_experiment(conditions)

# Run the agent
agent = CausalUncertaintyAgentLee2015()
alpha_hist, lr_hist = agent.run_experiment(schedule_df)
ratings = agent.current_ratings()
```

### 2. Causal Inference Task (MATLAB)

**Run experimental simulations:**

```matlab
% Generate random event sequences (first session)
event_random(subject, block_num)

% Use previous session data to influence current session
given_event_made(subject, BLOCK_NUM)

% Run main experimental session
causal_inference_exp('test', 1, 'pre')   % Practice session
causal_inference_exp('test', 1, 'main')  % Main session
```

See the [Causal inference task README](Causal%20inference%20task/README.md) for detailed usage instructions.

## Component Details

### Causal Uncertainty Model (Python)

Implements a Bayesian agent that:
- Maintains Dirichlet beliefs over causal strength for each stimulus
- Converts Dirichlet variance ("causal uncertainty") into adaptive one-shot learning rates via softmax
- Incorporates primacy and recency effects at the block level
- Can be fit to subject rating data via bounded optimization

**Key Features:**
- Task schedule generation (one-shot/incremental conditions)
- Trial-wise and block-wise parameter fitting
- Support for both "sole" (presence/absence) and "additive" (count) update modes
- Trajectory visualization and comparison
- Full compatibility with Lee et al. (2015) methodology

**Model Parameters:**
- `lr`: Base learning rate scaling Dirichlet α updates (default: 0.1)
- `primacy`: Extra weight for first stimulus in block (default: 0.36)
- `recency`: Extra weight for last stimulus in block (default: 0.37)
- `temperature`: Softmax temperature for OS learning rate (default: 255.0)
- `alpha0_init`: Initial Dirichlet α for each stimulus (default: 1.0)
- `update_mode`: "sole" or "additive" (default: "sole")

See the [causal uncertainty model README](causal%20uncertainty%20model/README.md) for detailed documentation.

### Causal Inference Task (MATLAB)

MATLAB scripts for experimental simulation with visual stimuli:
- **causal_inference_exp.m**: Main experimental session with visual stimuli, reward feedback, and multiple response types (hedonic ratings, causal attributions, Bayesian betting)
- **event_random.m**: Generates randomized stimulus and reward sequences for first session
- **given_event_made.m**: Uses previous session data for progressive/adaptive experiments

**Experimental Structure:**
- Up to 5 sessions per participant
- 8 blocks per session (2 practice blocks for pre-sessions)
- 5 trials per block, 5 cues per trial
- 3 images per block (2 common stimuli + 1 novel stimulus)
- Multiple response types: hedonic ratings, causal attribution ratings, triangular Bayesian betting

See the [Causal inference task README](Causal%20inference%20task/README.md) for detailed documentation.

## Usage Examples

### Python: Generate and Run a Task Schedule

```python
from task_lee2015 import OneShotIncrementalTaskLee2015
from model_lee2015 import CausalUncertaintyAgentLee2015

# Generate task schedule
task = OneShotIncrementalTaskLee2015(seed=42)
conditions = ["oneshot", "incremental", "oneshot", "incremental"]
schedule_df = task.generate_experiment(conditions)

# Run the agent
agent = CausalUncertaintyAgentLee2015(
    lr=0.1,
    primacy=0.36,
    recency=0.37,
    temperature=255.0,
    update_mode="additive"
)
alpha_hist, lr_hist = agent.run_experiment(schedule_df)
ratings = agent.current_ratings()

# Get trajectory DataFrame
traj_df = agent.get_trajectory_df(schedule_df)
```

### Python: Fit Model to Subject Data

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
    update_mode="additive",
    maxiter=200
)
print("Best-fit params:", fit_result["params"])

# Or fit both modes and compare
fit_both = agent.fit_to_subject_trialwise_both(
    schedule_df=schedule_df,
    subject_df=subject_df,
    cues_per_block=5,
    maxiter=200
)
print("Best mode:", fit_both["best_mode"])
```

### MATLAB: Run Experimental Session

```matlab
% Practice session (2 blocks, longer time limits)
causal_inference_exp('test', 1, 'pre');

% Main session 1 (8 blocks, normal timing)
causal_inference_exp('test', 1, 'main');

% Main session 2 (uses data from session 1)
causal_inference_exp('test', 2, 'main');
```

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

The key experimental manipulation is whether the rare co-occurrence happens in one block (oneshot) or is separated across blocks (incremental).

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

where τ is the temperature parameter. This ensures that stimuli with higher uncertainty receive higher learning rates.

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

> Lee SW, O'Doherty JP, Shimojo S (2015) Neural Computations Mediating One-Shot Learning in the Human Brain. *PLoS Biology*, 13(4): e1002137.

## License

This project is licensed under the MIT License.

## Additional Resources

- **Causal uncertainty model**: See [`causal uncertainty model/README.md`](causal%20uncertainty%20model/README.md) for detailed Python model documentation
- **Causal inference task**: See [`Causal inference task/README.md`](Causal%20inference%20task/README.md) for detailed MATLAB experimental protocol documentation
- **Experimental instructions**: See `Causal inference task/Instruction.pdf` for participant-facing experimental protocol details
- The examples in each folder demonstrate typical usage patterns

## Notes

- The Python and MATLAB implementations are independent and can be used separately
- The MATLAB scripts are designed for use with Psychtoolbox in experimental settings (including fMRI/MRI scanners)
- Random seeds are provided in the Causal inference task folder for reproducibility
- The Python model can be fit to data collected from the MATLAB experiment or used for simulation studies

## Contact and Support

For questions about:
- **Python model**: See the detailed README in `causal uncertainty model/`
- **MATLAB experiment**: See the detailed README in `Causal inference task/`
- **Experimental protocol**: See `Causal inference task/Instruction.pdf`
- **Original paper**: Lee et al. (2015) PLoS Biology
