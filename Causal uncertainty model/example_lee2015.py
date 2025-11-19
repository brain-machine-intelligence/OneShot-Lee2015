from task_lee2015 import OneShotIncrementalTaskLee2015
from model_lee2015 import CausalUncertaintyAgentLee2015
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def print_section(title, width=70):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title, width=70):
    """Print a formatted subsection header."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print("-" * width)


def format_params(params, labels=None):
    """Format parameter values for display."""
    if labels is None:
        labels = ["lr", "primacy", "recency", "tau"]
    formatted = []
    for label, val in zip(labels, params):
        formatted.append(f"{label}={val:.4f}")
    return ", ".join(formatted)

def plot_fitted_trajectory(traj_df, title="Fitted model trajectory"):
    """
    Plot the fitted model trajectory showing rating and one-shot learning rate over blocks.
    
    Parameters
    ----------
    traj_df : pandas.DataFrame
        DataFrame obtained from agent.get_trajectory_df(new_schedule_df).
        Expected columns: ['step', 'round_idx', 'block_idx', 'condition',
                          'stimulus', 'alpha', 'rating', 'os_lr']
    title : str
        Title for the plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # -------------------------
    # 1) rating trajectory
    # -------------------------
    ax = axes[0]
    for s in sorted(traj_df["stimulus"].unique()):
        df_s = traj_df[traj_df["stimulus"] == s]
        ax.plot(df_s["step"], df_s["rating"], label=f"stim {s}")
    ax.set_title(title + " — Rating trajectory")
    ax.set_ylabel("rating")
    ax.legend()

    # -------------------------
    # 2) OS learning rate trajectory
    # -------------------------
    ax = axes[1]
    for s in sorted(traj_df["stimulus"].unique()):
        df_s = traj_df[traj_df["stimulus"] == s]
        ax.plot(df_s["step"], df_s["os_lr"], label=f"stim {s}")
    ax.set_title(title + " — OS learning rate trajectory")
    ax.set_xlabel("block step")
    ax.set_ylabel("OS lr")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# 1. Generate schedule
# ==========================================================
print_section("1. GENERATE EXPERIMENTAL SCHEDULE")

task = OneShotIncrementalTaskLee2015(seed=42)
conditions = ["oneshot"] * 20 + ["incremental"] * 20
schedule_df = task.generate_experiment(conditions)

print(f"\nGenerated schedule: {len(conditions)} rounds")
print(f"  - Oneshot conditions: {conditions.count('oneshot')}")
print(f"  - Incremental conditions: {conditions.count('incremental')}")

print_subsection("Example: Round 0 Schedule (oneshot condition)")
task.pretty_print_round(schedule_df, 0)


# ==========================================================
# 2. Simulate trial-wise rating with true OS model (additive)
# ==========================================================
print_section("2. SIMULATE TRUE AGENT AND GENERATE SUBJECT DATA")

# Initialize true agent with known parameters
true_agent = CausalUncertaintyAgentLee2015(
    lr=0.1,
    primacy=0.36,
    recency=0.37,
    temperature=255.0,
    reward_high=1.0,
    reward_low=-1.0,
    update_mode="additive",   # Assume true subject uses additive mode
)

print("\nTrue agent parameters:")
print(f"  {format_params([0.1, 0.36, 0.37, 255.0])}")
print(f"  Update mode: additive")

# Block-by-block trajectory for the entire schedule
print("\nRunning true agent on schedule...")
true_agent.run_experiment(schedule_df, cues_per_block=5)
traj_true = true_agent.get_trajectory_df(schedule_df)

# Add noise to ratings to simulate subject data
print("Generating noisy subject ratings...")
rng = np.random.RandomState(0)
rows = []
for _, row in traj_true.iterrows():
    r = int(row["round_idx"])
    b = int(row["block_idx"])
    s = int(row["stimulus"])
    rating = float(row["rating"])

    noisy = rating + rng.normal(0.0, 0.02)
    noisy = max(noisy, 0.0)

    rows.append(
        {
            "round_idx": r,
            "block_idx": b,
            "stimulus": s,
            "subject_rating": noisy,
        }
    )

subject_df = pd.DataFrame(rows)

# Normalize rating per stimulus within each block (so sum equals 1)
subject_df["subject_rating"] = subject_df.groupby(
    ["round_idx", "block_idx"]
)["subject_rating"].transform(lambda x: x / (x.sum() + 1e-8))

print_subsection("Subject Ratings Sample (first 10 rows)")
print(subject_df.head(10).to_string(index=False))
print(f"\nTotal subject data points: {len(subject_df)}")


# ==========================================================
# 3. Trial-wise fitting for both additive and sole modes
# ==========================================================
print_section("3. FIT MODEL TO SUBJECT DATA")

agent_fit = CausalUncertaintyAgentLee2015()

print("\nFitting both 'sole' and 'additive' update modes...")
print("  (This may take a moment)")
fit_all = agent_fit.fit_to_subject_trialwise_both(
    schedule_df=schedule_df,
    subject_df=subject_df,
    cues_per_block=5,
    initial_params=(0.1, 0.36, 0.37, 255.0),
    maxiter=200,
    verbose=False,  # Set to True for optimization details
)

print_subsection("Fitting Results")
print(f"\nTrue mode:        additive")
print(f"Best fit mode:    {fit_all['best_mode']}")
print(f"Best loss (MSE):   {fit_all['best']['loss']:.6f}")

print("\nFitted parameters (best mode):")
best_params = fit_all["best"]["params"]
print(f"  {format_params(best_params)}")

print_subsection("Comparison: Sole vs Additive Modes")
sole_params = fit_all["sole"]["params"]
add_params = fit_all["additive"]["params"]
sole_loss = fit_all["sole"]["loss"]
add_loss = fit_all["additive"]["loss"]

print(f"\nSole mode:")
print(f"  Parameters: {format_params(sole_params)}")
print(f"  Loss (MSE):  {sole_loss:.6f}")

print(f"\nAdditive mode:")
print(f"  Parameters: {format_params(add_params)}")
print(f"  Loss (MSE):  {add_loss:.6f}")

if fit_all["best_mode"] == "additive":
    print(f"\n✓ Additive mode fits better (Δ = {sole_loss - add_loss:.6f})")
else:
    print(f"\n✓ Sole mode fits better (Δ = {add_loss - sole_loss:.6f})")


# ==========================================================
# 4. Give "new experiment" to fitted model and obtain ratings
# ==========================================================
print_section("4. SIMULATE FITTED MODEL ON NEW EXPERIMENT")

# 4-1. Extract fitted parameters and mode
best_mode = fit_all["best_mode"]
lr_fit, prim_fit, rec_fit, tau_fit = fit_all["best"]["params"]

print("\nFitted model configuration:")
print(f"  Update mode: {best_mode}")
print(f"  Parameters:  {format_params([lr_fit, prim_fit, rec_fit, tau_fit])}")

# 4-2. Create new experiment schedule
print("\nGenerating new experiment schedule...")
new_task = OneShotIncrementalTaskLee2015(seed=999)
new_conditions = ["oneshot"] * 10 + ["incremental"] * 10
new_schedule_df = new_task.generate_experiment(new_conditions)

print(f"  New schedule: {len(new_conditions)} rounds")
print(f"    - Oneshot: {new_conditions.count('oneshot')} rounds")
print(f"    - Incremental: {new_conditions.count('incremental')} rounds")

print_subsection("Example: New Round 0 Schedule")
new_task.pretty_print_round(new_schedule_df, 0)

# 4-3. Run entire new experiment with fitted model
print("\nRunning fitted model on new schedule...")
fitted_agent = CausalUncertaintyAgentLee2015(
    n_stim=3,
    lr=lr_fit,
    primacy=prim_fit,
    recency=rec_fit,
    temperature=tau_fit,
    alpha0_init=1.0,
    reward_high=1.0,
    reward_low=-1.0,   # Match settings used during fitting
    update_mode=best_mode,
)

fitted_agent.run_experiment(new_schedule_df, cues_per_block=5)
final_ratings = fitted_agent.current_ratings()

print_subsection("Final Ratings (Whole Session)")
print("\nStimulus ratings after full experiment:")
for s, rating in enumerate(final_ratings):
    print(f"  Stimulus {s}: {rating:.4f}")
print(f"  (Sum: {sum(final_ratings):.4f})")

# 4-4. Obtain ratings per round
print("\nComputing round-wise ratings...")
round_ratings = []  # Store (round_idx, [r0,r1,r2]) for each round

for r in sorted(new_schedule_df["round_idx"].unique()):
    sched_r = new_schedule_df[new_schedule_df["round_idx"] == r]

    # Create new agent for alpha reset per round
    round_agent = CausalUncertaintyAgentLee2015(
        n_stim=3,
        lr=lr_fit,
        primacy=prim_fit,
        recency=rec_fit,
        temperature=tau_fit,
        alpha0_init=1.0,
        reward_high=1.0,
        reward_low=-1.0,
        update_mode=best_mode,
    )
    round_agent.run_experiment(sched_r, cues_per_block=5)
    r_rating = round_agent.current_ratings()
    round_ratings.append((r, r_rating))

print_subsection("Round-wise Ratings (First 5 Rounds)")
print("\nRound | Stimulus 0 | Stimulus 1 | Stimulus 2 | Condition")
print("-" * 60)
for r, rr in round_ratings[:5]:
    condition = new_schedule_df[new_schedule_df["round_idx"] == r]["condition"].iloc[0]
    print(f"  {r:2d}  |   {rr[0]:.4f}   |   {rr[1]:.4f}   |   {rr[2]:.4f}   | {condition}")

# 4-5. Extract and display trajectory
traj_new = fitted_agent.get_trajectory_df(new_schedule_df)

print_subsection("Trajectory Summary")
print(f"\nTotal blocks in trajectory: {len(traj_new) // 3}")  # 3 stimuli per block
print(f"Total steps: {traj_new['step'].max() + 1}")

print("\nTrajectory sample (first 10 rows):")
print(traj_new.head(10).to_string(index=False))

# Plot trajectory
print("\n" + "=" * 70)
print("  Generating trajectory plots...")
print("=" * 70)
plot_fitted_trajectory(
    traj_new,
    title=f"Trajectory of fitted agent (mode={best_mode})"
)

print("\n" + "=" * 70)
print("  Analysis complete!")
print("=" * 70)
