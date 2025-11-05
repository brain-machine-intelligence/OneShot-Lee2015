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
        rewards = [1 if outcome_assignment[idx] == 'novel' else -1 for idx in range(self.cues_per_round)]

        round_events = []
        for s_idx, stimulus in enumerate(cues):
            primacy_eff = 1.0 if s_idx == 0 else 0.0
            recency_eff = 1.0 if s_idx == len(cues) - 1 else 0.0
            outcome = outcome_assignment[s_idx]
            reward = rewards[s_idx]
            prior_mean = self.mean.loc[stimulus, outcome]
            prior_var = self.var.loc[stimulus, outcome]
            likelihood_var = 0.5

            posterior_var = 1 / (1 / prior_var + 1 / likelihood_var)
            posterior_mean = posterior_var * (
                (prior_mean / prior_var) + (reward / likelihood_var)
            )
            self.mean.loc[stimulus, outcome] = posterior_mean
            self.var.loc[stimulus, outcome] = posterior_var

            lr_key = (stimulus, outcome)
            lr_dict = self.compute_learning_rates()
            lr = lr_dict[lr_key]

            d_alpha = (
                lr *
                reward *
                (1.0 + self.primacy * primacy_eff + self.recency * recency_eff)
            )
            self.mean.loc[stimulus, outcome] += self.lr_base * d_alpha
            self.mean.loc[stimulus, outcome] = max(self.mean.loc[stimulus, outcome], 0.0)

            round_events.append({
                'stimulus': stimulus,
                'outcome': outcome,
                'novelty': True if s_idx == novel_cue_idx else False,
                'reward': reward,
                'primacy': primacy_eff,
                'recency': recency_eff,
                'lr': lr,
                'mean': self.mean.loc[stimulus, outcome],
                'var': self.var.loc[stimulus, outcome]
            })

        lr_vals = [e['lr'] for e in round_events]
        lr_90 = np.percentile(lr_vals, 90)
        for e in round_events:
            e['tag'] = 'OS' if e['lr'] >= lr_90 else 'IC'

        self.history.append(round_events)
        return round_events

    def simulate(self, round_types=None):
        if round_types is None:
            round_types = ['type1'] * int(self.rounds / 2) + ['type2'] * int(self.rounds / 2)
            np.random.shuffle(round_types)
        for rtype in round_types:
            self.run_round(rtype)

    def output_dataframe(self):
        rows = []
        for r_idx, round_event in enumerate(self.history):
            for e in round_event:
                row = e.copy()
                row['round_idx'] = r_idx
                rows.append(row)
        return pd.DataFrame(rows)

    def fit_to_behavior(self, subject_data, initial_params=None):
        if initial_params is None:
            initial_params = [self.lr_base, self.lr_temp, self.primacy, self.recency]

        def loss(params):
            lr_base, lr_temp, primacy, recency = params
            sim = CausalUncertaintyModel(
                stim_types=self.stim_types,
                cues_per_round=self.cues_per_round,
                rounds=self.rounds,
                lr_base=lr_base,
                lr_temp=lr_temp,
                primacy=primacy,
                recency=recency
            )
            sim.simulate(round_types=list(subject_data['round_type']))
            model_df = sim.output_dataframe()
            merged = pd.merge(
                subject_data,
                model_df,
                on=['round_idx', 'stimulus', 'outcome'],
                suffixes=('_subj', '_model')
            )
            mse = np.mean((merged['subject_rating'] - merged['mean']) ** 2)
            return mse

        result = minimize(
            loss,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'disp': True}
        )
        self.set_params(
            lr_base=result.x[0], lr_temp=result.x[1], primacy=result.x[2], recency=result.x[3]
        )
        return dict(params=result.x, loss=result.fun, result=result)
