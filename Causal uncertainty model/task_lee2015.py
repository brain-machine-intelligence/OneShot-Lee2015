# task_lee2015.py
import numpy as np
import pandas as pd


class OneShotIncrementalTaskLee2015:
    """
    Stimulus schedule generator for the one-shot / incremental task in Lee et al. (2015).

    This class creates *task structures* (stimulus / outcome sequences) that
    mimic the behavioral paradigm in the paper. The generated schedule can be
    fed directly into `CausalUncertaintyAgentLee2015` in `model_lee2015.py`.

    Conventions
    ----------
    - Stimuli: 3 discrete cue types, encoded as integers {0, 1, 2}
    - Outcomes: 2 discrete outcomes, encoded as integers {0, 1}
    - One round:
        * `blocks_per_round` blocks
        * `cues_per_block` cue presentations per block
        * 1 outcome per block (shared by all cues in that block)
    - Frequency constraints *within* one round:
        * `cue_counts`:       {0: 16, 1: 8, 2: 1}
        * `outcome_counts`:   {0: 4, 1: 1}
    - Condition:
        * `"oneshot"`     : rare cue & rare outcome occur in the same block
        * `"incremental"` : rare cue & rare outcome occur in *different* blocks

    The goal is to construct schedules that satisfy these marginal constraints
    while enforcing the specified relation between the rare cue and outcome.
    """

    def __init__(
        self,
        cue_counts=None,
        outcome_counts=None,
        blocks_per_round=5,
        cues_per_block=5,
        seed=None,
    ):
        """
        Parameters
        ----------
        cue_counts : dict or None
            Mapping from stimulus ID → number of appearances per round.
            If `None`, defaults to {0: 16, 1: 8, 2: 1}.
        outcome_counts : dict or None
            Mapping from outcome ID → number of appearances per round.
            If `None`, defaults to {0: 4, 1: 1}.
        blocks_per_round : int
            Number of blocks in one round.
        cues_per_block : int
            Number of cue presentations per block.
        seed : int or None
            Seed for NumPy's RandomState for reproducibility.
        """
        if cue_counts is None:
            cue_counts = {0: 16, 1: 8, 2: 1}
        if outcome_counts is None:
            outcome_counts = {0: 4, 1: 1}

        self.cue_counts = cue_counts
        self.outcome_counts = outcome_counts
        self.blocks_per_round = blocks_per_round
        self.cues_per_block = cues_per_block

        # Sanity‑check that the requested counts are internally consistent.
        total_cues = sum(cue_counts.values())
        total_outcomes = sum(outcome_counts.values())

        assert total_cues == blocks_per_round * cues_per_block, (
            "cue_counts must sum to blocks_per_round * cues_per_block."
        )
        assert total_outcomes == blocks_per_round, (
            "outcome_counts must sum to blocks_per_round."
        )

        self.rng = np.random.RandomState(seed)

        # Automatically pick rare stimulus / rare outcome
        # (those with the smallest counts in the provided dictionaries).
        self.rare_stim = min(cue_counts, key=cue_counts.get)
        self.rare_outcome = min(outcome_counts, key=outcome_counts.get)

    def generate_round(self, round_idx=0, condition="oneshot"):
        """
        Generate the schedule for a single round and return it as a DataFrame.

        Parameters
        ----------
        round_idx : int
            Identifier for this round (stored in the output DataFrame).
        condition : {"oneshot", "incremental"}
            Whether the rare cue and rare outcome should co‑occur in the same
            block (`"oneshot"`) or in different blocks (`"incremental"`).

        Returns
        -------
        pandas.DataFrame
            Columns:
                - `round_idx`
                - `block_idx` (0‑based)
                - `pos_in_block` (0‑based position of cue within its block)
                - `condition`
                - `stimulus` (cue ID)
                - `outcome` (block outcome)
        """
        # 1) Build cue sequence (length = blocks_per_round * cues_per_block)
        cue_seq = []
        for stim, n in self.cue_counts.items():
            cue_seq.extend([stim] * n)
        cue_seq = np.array(cue_seq)
        self.rng.shuffle(cue_seq)

        # 2) Build outcome sequence (length = blocks_per_round)
        outcome_seq = []
        for o, n in self.outcome_counts.items():
            outcome_seq.extend([o] * n)
        outcome_seq = np.array(outcome_seq)
        self.rng.shuffle(outcome_seq)

        # 3) Find the block indices of the rare cue and rare outcome.
        #    We first compute, for every cue presentation, which block it
        #    belongs to (0..blocks_per_round-1).
        block_indices = np.repeat(
            np.arange(self.blocks_per_round),
            self.cues_per_block
        )
        rare_stim_trial_idx = np.where(cue_seq == self.rare_stim)[0][0]
        rare_stim_block = block_indices[rare_stim_trial_idx]

        rare_outcome_block = np.where(outcome_seq == self.rare_outcome)[0][0]

        # 4) Adjust rare_stim_block and rare_outcome_block depending on condition.
        #    This is done *only* by permuting the outcome sequence, so the
        #    marginal frequencies of cues and outcomes remain unchanged.
        if condition == "oneshot":
            # Make rare_stim_block == rare_outcome_block by swapping outcomes if needed.
            if rare_stim_block != rare_outcome_block:
                tmp = outcome_seq[rare_stim_block]
                outcome_seq[rare_stim_block] = outcome_seq[rare_outcome_block]
                outcome_seq[rare_outcome_block] = tmp

        elif condition == "incremental":
            # Make rare_stim_block != rare_outcome_block by adjusting outcomes if needed.
            if rare_stim_block == rare_outcome_block:
                candidates = [b for b in range(self.blocks_per_round)
                              if b != rare_stim_block]
                new_block = self.rng.choice(candidates)
                tmp = outcome_seq[new_block]
                outcome_seq[new_block] = outcome_seq[rare_outcome_block]
                outcome_seq[rare_outcome_block] = tmp
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # 5) Expand from block‑level outcomes to cue‑level rows and build DataFrame.
        rows = []
        for b in range(self.blocks_per_round):
            start = b * self.cues_per_block
            end = start + self.cues_per_block
            cues_block = cue_seq[start:end]
            outcome_b = outcome_seq[b]

            for pos, stim in enumerate(cues_block):
                rows.append({
                    "round_idx": round_idx,
                    "block_idx": b,
                    "pos_in_block": pos,
                    "condition": condition,
                    "stimulus": int(stim),
                    "outcome": int(outcome_b),
                })

        return pd.DataFrame(rows)

    def generate_experiment(self, conditions):
        """
        conditions: list of length n_rounds
            e.g., ['oneshot', 'incremental', 'oneshot', ...]
        """
        frames = []
        for r_idx, cond in enumerate(conditions):
            df_round = self.generate_round(round_idx=r_idx, condition=cond)
            frames.append(df_round)
        return pd.concat(frames, ignore_index=True)

    def pretty_print_round(self, schedule_df, round_idx):
        """
        Nicely print a single round in a 5×6 block table (5 cues + 1 outcome).
        """
        df = schedule_df.query("round_idx == @round_idx")
        print(f"\n=== Round {round_idx} ({df['condition'].iloc[0]}) ===")

        # Sort by block
        for b in range(self.blocks_per_round):
            dfb = df.query("block_idx == @b").sort_values("pos_in_block")
            cues = dfb["stimulus"].tolist()  # [s0, s1, s2, s3, s4]
            outcome = dfb["outcome"].iloc[0]  # block outcome
            cues_str = " ".join(str(c) for c in cues)
            print(f"Block {b}: {cues_str} | outcome {outcome}")
