# One-Shot Learning: Lee et al. (2015) Implementation

This repository contains implementations of the one-shot learning model and experimental paradigm from:

> Lee SW, O'Doherty JP, Shimojo S (2015)  
> Neural Computations Mediating One-Shot Learning in the Human Brain.  
> *PLoS Biology*, 13(4): e1002137.

The project includes computational models, task generators, and experimental simulation tools for studying one-shot vs. incremental causal learning.

## Project Overview

This repository is organized into three main components:

1. **Causal Inference Task** (MATLAB) - Experimental simulation scripts for running causal inference experiments
2. **Causal Learning Model** (Python) - Core one-shot learning model implementation
3. **Causal Uncertainty Model** (Python) - Bayesian causal uncertainty agent with Dirichlet priors

## Repository Structure

```
OneShot-Lee2015-main/
├── Causal inference task/          # MATLAB experimental simulation
│   ├── causal_inference_exp.m      # Main experimental session script
│   ├── event_random.m              # Random event sequence generator
│   ├── given_event_made.m          # Progressive event generator
│   ├── Instruction.pdf             # Experimental instructions
│   └── seed.vol1.egg, seed.vol2.egg # Random seed files
│
├── Causal learning model/          # Core one-shot model
│   ├── Oneshot.py                  # Main Oneshot class
│   ├── Oneshot_sangwan.py          # Core model implementation
│   └── gen_exp.py                  # Example usage script
│
└── causal uncertainty model/       # Bayesian causal uncertainty agent
    ├── model_lee2015.py            # CausalUncertaintyAgentLee2015 class
    ├── task_lee2015.py             # Task schedule generator
    ├── example_lee2015.py           # Basic usage example
    └── example_lee2015_trajectories.py  # Trajectory comparison example
```

## Installation

### Python Components

The Python components require standard scientific libraries:

```bash
pip install numpy pandas scipy matplotlib
```

### MATLAB Components

The MATLAB scripts require:
- MATLAB (R2014b or later recommended)
- [Psychtoolbox-3](http://psychtoolbox.org/) for visual stimulus presentation

## Quick Start

### 1. Causal Learning Model (Python)

Run the basic example to see one-shot vs. incremental learning:

```bash
cd "Causal learning model"
python gen_exp.py
```

This generates:
- An incremental episode (reward dissociated from novel stimulus)
- A one-shot episode (reward tied to novel stimulus)

### 2. Causal Uncertainty Model (Python)

**Basic simulation and parameter fitting:**

```bash
cd "causal uncertainty model"
python example_lee2015.py
```

This will:
- Generate a one-shot schedule (10 rounds)
- Run the causal uncertainty agent
- Display final Dirichlet α values and causal ratings
- Fit model parameters to dummy subject data

**Compare one-shot vs. incremental trajectories:**

```bash
python example_lee2015_trajectories.py
```

This generates plots showing:
- Causal ratings over blocks
- One-shot learning rates over blocks
- Comparison between pure one-shot and pure incremental conditions

### 3. Causal Inference Task (MATLAB)

Run experimental simulations in MATLAB:

```matlab
% Generate random event sequences
event_random(subject, block_num)

% Use previous session data to influence current session
given_event_made(subject, BLOCK_NUM)

% Run main experimental session
causal_inference_exp
```

## Component Details

### Causal Learning Model

Implements the core one-shot learning mechanism with:
- Stimulus and cue vector representations
- Primacy and recency effects
- Adaptive learning rates based on stimulus novelty

**Key Features:**
- Configurable feature dimensions
- Customizable stimulus and cue ratios
- Episode generation for both one-shot and incremental conditions

### Causal Uncertainty Model

Implements a Bayesian agent that:
- Maintains Dirichlet beliefs over causal strength for each stimulus
- Converts Dirichlet variance ("causal uncertainty") into adaptive one-shot learning rates
- Incorporates primacy and recency effects at the block level
- Can be fit to subject rating data via bounded optimization

**Key Features:**
- Task schedule generation (one-shot/incremental conditions)
- Trial-wise and block-wise parameter fitting
- Trajectory visualization and comparison
- Full compatibility with Lee et al. (2015) methodology

### Causal Inference Task

MATLAB scripts for experimental simulation:
- **causal_inference_exp.m**: Main experimental session with visual stimuli
- **event_random.m**: Generates randomized stimulus and reward sequences
- **given_event_made.m**: Uses previous session data for progressive experiments

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
agent = CausalUncertaintyAgentLee2015()
alpha_hist, lr_hist = agent.run_experiment(schedule_df)
ratings = agent.current_ratings()
```

### Python: Fit Model to Subject Data

```python
import pandas as pd

subject_df = pd.DataFrame({
    "stimulus": [0, 1, 2],
    "subject_rating": [0.3, 0.6, 0.1]
})

fit_result = agent.fit_to_subject(schedule_df, subject_df, maxiter=200)
print("Best-fit params (lr, primacy, recency):", fit_result["params"])
```

## Citation

If you use this code in your work, please cite:

> Lee SW, O'Doherty JP, Shimojo S (2015) Neural Computations Mediating One-Shot Learning in the Human Brain. *PLoS Biology*, 13(4): e1002137.

## License

This project is licensed under the MIT License.



