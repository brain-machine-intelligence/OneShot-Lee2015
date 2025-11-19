# model_lee2015.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class CausalUncertaintyAgentLee2015:
    """
    Causal uncertainty model based on a Dirichlet prior as in Lee et al. (2015).

    The agent maintains a Dirichlet belief over the causal strength of each
    stimulus. Uncertainty in this Dirichlet distribution is used to derive
    an *adaptive one-shot learning rate* for each stimulus.

    Key ideas
    ---------
    - State:
        * `alphas`: Dirichlet concentration parameters α_s for each stimulus type.
    - Causal uncertainty (for each stimulus s):
        * Var[θ_s] = α_s (α_0 − α_s) / (α_0² (α_0 + 1)),
          where α_0 = Σ_s α_s.
    - One-shot (OS) learning rate:
        * lr_s = softmax(temperature × CU_s),
          where CU_s is the causal uncertainty for stimulus s.
    - Update rule (per block):
        * Δα_s ∝ lr_s × reward ×
          (saliency_s + primacy × I_primacy(s) + recency × I_recency(s))

    This file includes a trial-wise fitting method
    `fit_to_subject_trialwise` that estimates (lr, primacy, recency, temperature)
    by minimising trial-by-trial MSE between model ratings and subject ratings,
    matching the description in Lee et al. (2015).
    """

    def __init__(
        self,
        n_stim=3,
        lr=0.1,
        primacy=0.36,
        recency=0.37,
        temperature=255.0,
        alpha0_init=1.0,
        reward_high=1.0,
        reward_low=-1.0,
        update_mode="sole",
    ):
        """
        Parameters
        ----------
        n_stim : int
            Number of distinct stimuli (cue types).
        lr : float
            Base learning rate scaling the change in Dirichlet α.
        primacy : float
            Extra weight for stimuli at the beginning of each block.
        recency : float
            Extra weight for stimuli at the end of each block.
        temperature : float
            Softmax temperature τ for mapping causal uncertainty to OS learning rate.
        alpha0_init : float
            Initial Dirichlet α value for each stimulus (symmetric prior).
        reward_high : float
            Reward value used when outcome == 1.
        reward_low : float
            Reward value used when outcome == 0 (typically 0.0 or -1.0).
        """
        self.n_stim = n_stim
        self.param_lr = lr
        self.param_primacy = primacy
        self.param_recency = recency
        self.temperature = temperature
        self.alpha0_init = alpha0_init

        # Set reward values corresponding to outcome 1 / 0
        self.reward_high = reward_high
        self.reward_low = reward_low

        self.update_mode = update_mode

        self.reset()

    # ------------------------------------------------------
    # State reset
    # ------------------------------------------------------
    def reset(self):
        """
        Reset internal state to the initial Dirichlet prior.

        This clears any history of α and learning rates and is automatically
        called at the beginning of `run_experiment`.
        """
        # Initial Dirichlet α (symmetric prior across stimuli).
        self.alphas = np.ones(self.n_stim) * self.alpha0_init
        # Histories are recorded at *block* resolution:
        #   - `alpha_history[0]` is the initial value (before any blocks),
        #   - `alpha_history[k]` (k ≥ 1) is after the k-th block update.
        self.alpha_history = [self.alphas.copy()]
        self.lr_history = []        # step k = OS learning rate used at block k
        self.block_keys = []        # (round_idx, block_idx) identifiers per block

    # ------------------------------------------------------
    # Dirichlet-based causal uncertainty → OS learning rate
    # ------------------------------------------------------
    def _compute_cu(self, alphas):
        """
        Compute causal uncertainty (variance) for each stimulus under a
        Dirichlet-multinomial model.

        Parameters
        ----------
        alphas : array_like, shape (n_stim,)
            Current Dirichlet concentration parameters for each stimulus.

        Returns
        -------
        np.ndarray
            Per-stimulus causal uncertainty values (same shape as `alphas`).
        """
        # Ensure non-negative values to keep the Dirichlet interpretation.
        alphas = np.maximum(alphas, 0.0)
        alpha0 = np.sum(alphas) + 1e-8
        cu = []
        for a in alphas:
            cu_s = (a * (alpha0 - a)) / ((alpha0 ** 2) * (alpha0 + 1.0))
            cu.append(cu_s)
        return np.array(cu)

    def _compute_os_lr(self, alphas):
        """
        Map causal uncertainty values to a one-shot learning rate via softmax.

        Parameters
        ----------
        alphas : array_like, shape (n_stim,)
            Current Dirichlet concentration parameters.

        Returns
        -------
        np.ndarray
            OS learning rate vector of length `n_stim` that sums to 1.
        """
        cu = self._compute_cu(alphas)
        logits = self.temperature * cu
        # Numerically stable softmax: subtract max(logits) before exponentiating.
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        lr = exp_l / np.sum(exp_l)
        return lr

    # ------------------------------------------------------
    # Compute x_i: sole vs additive
    # ------------------------------------------------------
    def _get_x(self, stims_block):
        """
        Compute the presence/count indicator x_i for each stimulus in a block.
        
        Parameters
        ----------
        stims_block : array_like
            Array of stimulus IDs in the block (length = cues_per_block).
        
        Returns
        -------
        np.ndarray
            Vector of length n_stim. Definition depends on update_mode:
            - "sole": 1 if stimulus appears at least once in block, else 0
            - "additive": count of how many times stimulus appears in block
        """
        if self.update_mode == "additive":
            # Additive effect: count occurrences
            return np.array(
                [(stims_block == s).sum() for s in range(self.n_stim)],
                dtype=float,
            )
        else:
            # Default: sole effect (presence/absence)
            return np.array(
                [1.0 if np.any(stims_block == s) else 0.0 for s in range(self.n_stim)],
                dtype=float,
            )


    # ------------------------------------------------------
    # Update for a single block (5 cues + 1 outcome)
    # ------------------------------------------------------
    def update_block(self, stims_block, outcome):
        """
        Update Dirichlet beliefs using the stimuli shown in a single block.

        Additive/Sole effect:
          x_i = _get_x(stims_block)[i]
        """
        alphas = self.alphas

        # 1) Compute x_i (additive or sole)
        x = self._get_x(stims_block)  # length n_stim

        # 2) Primacy / recency indicators
        first_stim = stims_block[0]
        last_stim = stims_block[-1]
        primacy_ind = np.zeros(self.n_stim)
        recency_ind = np.zeros(self.n_stim)
        primacy_ind[first_stim] = 1.0
        recency_ind[last_stim] = 1.0

        # 3) OS learning rate
        os_lr = self._compute_os_lr(alphas)

        # 4) Reward value (if using +1 / -1 structure, reward_low=-1.0)
        reward = self.reward_high if outcome == 1 else self.reward_low

        # 5) Update Dirichlet α
        for s in range(self.n_stim):
            d_alpha = os_lr[s] * reward * (
                    x[s]
                    + self.param_primacy * primacy_ind[s]
                    + self.param_recency * recency_ind[s]
            )
            alphas[s] += self.param_lr * d_alpha
            alphas[s] = max(alphas[s], 0.0)

        self.alphas = alphas
        self.alpha_history.append(self.alphas.copy())
        self.lr_history.append(os_lr.copy())

    # ------------------------------------------------------
    # Run the entire task given a schedule DataFrame
    # ------------------------------------------------------
    def run_experiment(self, schedule_df, cues_per_block=5):
        """
        Run the model on a full schedule of stimuli and outcomes.

        Parameters
        ----------
        schedule_df : pandas.DataFrame
            Must contain the columns
            ['round_idx', 'block_idx', 'pos_in_block',
             'condition', 'stimulus', 'outcome'].
            Typically produced by `OneShotIncrementalTaskLee2015`.
        cues_per_block : int
            Number of cues per block (used for sanity checking).

        Returns
        -------
        alpha_arr : np.ndarray, shape (n_blocks + 1, n_stim)
            History of Dirichlet α values (including initial prior).
        lr_arr : np.ndarray or None, shape (n_blocks, n_stim)
            History of OS learning rates per block (or `None` if no blocks).

        Notes
        -----
        This method clears any existing history by calling `reset()` at the
        start, then processes blocks in chronological order.
        """
        self.reset()
        schedule_df = schedule_df.sort_values(
            ["round_idx", "block_idx", "pos_in_block"]
        ).reset_index(drop=True)

        self.cues_per_block = cues_per_block
        self.block_keys = []

        for (round_idx, block_idx), df_block in schedule_df.groupby(
            ["round_idx", "block_idx"]
        ):
            stims_block = df_block["stimulus"].values
            outcome = df_block["outcome"].iloc[0]
            assert len(stims_block) == cues_per_block
            # Record block key (for reconstructing trajectories later)
            self.block_keys.append((int(round_idx), int(block_idx)))
            self.update_block(stims_block, outcome)

        # Return histories as arrays (can be used as-is)
        alpha_arr = np.vstack(self.alpha_history)  # shape: (n_blocks + 1, n_stim)
        lr_arr = np.vstack(self.lr_history) if len(self.lr_history) > 0 else None

        return alpha_arr, lr_arr

    # ------------------------------------------------------
    # Convert current α to causal ratings (normalized)
    # ------------------------------------------------------
    def current_ratings(self):
        """
        Convert current Dirichlet α values into normalized causal ratings.

        Returns
        -------
        np.ndarray
            Vector of length `n_stim` whose entries sum to 1.
        """
        alphas = np.maximum(self.alphas, 0.0)
        alpha0 = np.sum(alphas) + 1e-8
        return alphas / alpha0

    # ------------------------------------------------------
    # Export block-wise trajectory as a DataFrame
    # ------------------------------------------------------
    def get_trajectory_df(self, schedule_df):
        """
        Build a block-wise trajectory DataFrame of beliefs and learning rates.

        This should be called *after* `run_experiment`.

        For each block and each stimulus, the returned DataFrame contains:
            - round_idx, block_idx, condition
            - stimulus (0, 1, 2, ...)
            - alpha (after the block update)
            - rating (alpha / sum)
            - os_lr (OS learning rate used for that block)
        """
        # Mapping from (block) → condition
        cond_map = (
            schedule_df
            .drop_duplicates(subset=["round_idx", "block_idx"])
            .set_index(["round_idx", "block_idx"])["condition"]
            .to_dict()
        )

        rows = []
        for step, (r_idx, b_idx) in enumerate(self.block_keys):
            # lr_history[step]: OS learning rate used in this block
            os_lr = self.lr_history[step]
            # alpha_history[step + 1]: α after this block was updated
            alphas = self.alpha_history[step + 1]
            alphas = np.maximum(alphas, 0.0)
            alpha0 = np.sum(alphas) + 1e-8
            ratings = alphas / alpha0

            condition = cond_map[(r_idx, b_idx)]

            for s in range(self.n_stim):
                rows.append({
                    "step": step,              # global block index (0..N_blocks-1)
                    "round_idx": r_idx,
                    "block_idx": b_idx,
                    "condition": condition,
                    "stimulus": s,
                    "alpha": alphas[s],
                    "rating": ratings[s],
                    "os_lr": os_lr[s],
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------
    # Fit to subject: trial-wise (block-wise) ratings including τ
    # ------------------------------------------------------
    def fit_to_subject_trialwise(
            self,
            schedule_df,
            subject_df,
            cues_per_block=5,
            initial_params=None,
            maxiter=200,
            verbose=True,
            update_mode=None,
    ):
        """
        Trial-by-trial (block-wise) fitting with τ (temperature) included.

        Parameters
        ----------
        schedule_df : DataFrame
            Full schedule with columns
            ['round_idx', 'block_idx', 'pos_in_block',
             'condition', 'stimulus', 'outcome'].

        subject_df : DataFrame
            Trial-wise ratings.
            One row per (round_idx, block_idx, stimulus):
                ['round_idx', 'block_idx', 'stimulus', 'subject_rating']

        update_mode : {"sole", "additive"} or None
            - "sole": x_i = presence (0/1)
            - "additive": x_i = count (number of appearances in block)
            - None: use self.update_mode as is

        Returns
        -------
        dict with keys:
            "params": [lr, primacy, recency, temperature],
            "loss":   MSE over all (round, block, stim),
            "result": OptimizeResult,
            "mode":   update_mode used
        """
        if update_mode is None:
            update_mode = self.update_mode  # Use mode set in self if available

        if initial_params is None:
            initial_params = [
                self.param_lr,
                self.param_primacy,
                self.param_recency,
                self.temperature,
            ]

        def objective(params):
            lr, primacy, recency, temperature = params

            # Minimum stability constraints
            if lr <= 0 or temperature <= 0:
                return 1e6

            tmp_agent = CausalUncertaintyAgentLee2015(
                n_stim=self.n_stim,
                lr=lr,
                primacy=primacy,
                recency=recency,
                temperature=temperature,
                alpha0_init=self.alpha0_init,
                reward_high=self.reward_high,
                reward_low=self.reward_low,
                update_mode=update_mode,  # ★ Pass sole / additive here
            )

            tmp_agent.run_experiment(schedule_df, cues_per_block=cues_per_block)
            traj_df = tmp_agent.get_trajectory_df(schedule_df)

            model_df = traj_df[["round_idx", "block_idx", "stimulus", "rating"]]

            merged = pd.merge(
                subject_df,
                model_df,
                on=["round_idx", "block_idx", "stimulus"],
                how="inner",
                suffixes=("_subj", "_model"),
            )

            if len(merged) == 0:
                return 1e6

            mse = float(
                np.mean((merged["subject_rating"] - merged["rating"]) ** 2)
            )
            return mse

        # Paper used Nelder–Mead, but here we use stable L-BFGS-B + bounds
        bounds = [
            (1e-4, 3.0),  # lr
            (0.0, 3.0),  # primacy
            (0.0, 3.0),  # recency
            (1.0, 500.0),  # temperature τ
        ]

        result = minimize(
            objective,
            x0=np.array(initial_params, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "disp": verbose},
        )

        self.param_lr, self.param_primacy, self.param_recency, self.temperature = result.x

        return {
            "params": result.x,
            "loss": result.fun,
            "result": result,
            "mode": update_mode,
        }

    # ------------------------------------------------------
    # Fit both additive & sole modes trial-wise and compare
    # ------------------------------------------------------
    def fit_to_subject_trialwise_both(
        self,
        schedule_df,
        subject_df,
        cues_per_block=5,
        initial_params=None,
        maxiter=200,
        verbose=True,
    ):
        """
        Run trial-wise fitting for both additive and sole versions,
        then select the one that fits better.

        Returns
        -------
        dict with keys:
            "best_mode": "sole" or "additive"
            "best":      best fit result dict
            "sole":      sole mode fit result
            "additive":  additive mode fit result
        """
        results = {}

        for mode in ["sole", "additive"]:
            if verbose:
                print(f"\n[fit_to_subject_trialwise] mode = {mode}")
            res = self.fit_to_subject_trialwise(
                schedule_df=schedule_df,
                subject_df=subject_df,
                cues_per_block=cues_per_block,
                initial_params=initial_params,
                maxiter=maxiter,
                verbose=verbose,
                update_mode=mode,
            )
            results[mode] = res

        loss_sole = results["sole"]["loss"]
        loss_add = results["additive"]["loss"]

        if loss_sole <= loss_add:
            best_mode = "sole"
        else:
            best_mode = "additive"

        best_res = results[best_mode]

        # Update agent internal parameters to match best fit
        self.param_lr, self.param_primacy, self.param_recency, self.temperature = best_res["params"]
        self.update_mode = best_mode

        return {
            "best_mode": best_mode,
            "best": best_res,
            "sole": results["sole"],
            "additive": results["additive"],
        }
