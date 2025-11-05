# Causal Uncertainty Model

**Python implementation of the Bayesian causal uncertainty agent from:**

Lee SW, O’Doherty JP, Shimojo S (2015) Neural Computations Mediating One-Shot Learning in the Human Brain. PLoS Biol 13(4): e1002137.

## Features

- Bayesian updating of stimulus-outcome causal strength & uncertainty
- Event-by-event learning rate control (softmax over uncertainty)
- Primacy/recency behavioral effects (fittable per subject)
- Simulation of experimental task (novelty pairings, OS/IC tagging)
- Behavioral rating parameter fitting interface

## Usage

See `example_fit.py` for simulation, parameter fitting, and output.

## Citation

If used, please cite:
Lee SW, O’Doherty JP, Shimojo S (2015) PLoS Biol 13(4): e1002137.

---

For support, extensions, or fitting guidance: [Your Contact Information]
```

***

### causal_uncertainty.py

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class CausalUncertaintyModel:
    """
    Bayesian causal uncertainty agent as described in:
    Lee SW, O’Doherty JP, Shimojo S (2015), PLoS Biol 13(4): e1002137.
    Combines Bayesian updates, learning rate control via uncertainty, and primacy/recency behavioral effects.
    Usable for experiment simulation and behavioral fitting.
    """
    def __init__(
        self,
        stim_types=3,
        cues_per_round=5,
        rounds=40,
        lr_base=0.1,
        lr_temp=3.0,
        primacy=0.36,
        recency=0.36
    ):
        self.stim_types = stim_types
        self.cues_per_round = cues_per_round
        self.rounds = rounds
        self.lr_base = lr_base
        self.lr_temp = lr_temp
        self.primacy = primacy
        self.recency = recency

        self.stimuli = list(range(stim_types))
        self.outcomes = ['non-novel', 'novel']
        self.mean = pd.DataFrame(0.5, index=self.stimuli, columns=self.outcomes)
        self.var = pd.DataFrame(1.0, index=self.stimuli, columns=self.outcomes)
        self.history = []

    def set_params(self, lr_base=None, lr_temp=None, primacy=None, recency=None):
        if lr_base is not None: self.lr_base = lr_base
        if lr_temp is not None: self.lr_temp = lr_temp
        if primacy is not None: self.primacy = primacy
        if recency is not None: self.recency = recency

    def get_params(self):
        return {
            "lr_base": self.lr_base,
            "lr_temp": self.lr_temp,
            "primacy": self.primacy,
            "recency": self.recency
        }

    def compute_learning_rates(self):
        uncertainties = self.var.values.flatten()
        exp_unc = np.exp(self.lr_temp * uncertainties)
        lr = exp_unc / exp_unc.sum()
        keys = [(s, o) for s in self.stimuli for o in self.outcomes]
        return pd.Series(lr, index=pd.MultiIndex.from_tuples(keys))

    def run_round(self, round_type='type1'):
        cues = np.random.choice(self.stimuli, self.cues_per_round, replace=False)
        novel_cue_idx = np.random.choice(self.cues_per_round)
        outcome_assignment = ['non-novel'] * self.cues_per_round
        outcome_assignment[novel_cue_idx] = 'novel'
        rewards = [1 if outcome_assignment[idx] == 'novel' else -1 for idx in
