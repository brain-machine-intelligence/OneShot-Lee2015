# Causal Inference Experiment - MATLAB Implementation

This folder contains MATLAB scripts for running the causal inference one-shot learning experiment based on Lee et al. (2015). The scripts generate experimental schedules, present visual stimuli using Psychtoolbox, and collect participant responses including hedonic ratings, causal attributions, and Bayesian probability estimates.

## Citation

> Lee SW, O'Doherty JP, Shimojo S (2015) Neural Computations Mediating One-Shot Learning in the Human Brain. *PLoS Biology*, 13(4): e1002137.

## Overview

This implementation provides a complete experimental framework for studying one-shot vs. incremental causal learning. The experiment presents participants with visual cues of varying novelty, delivers reward outcomes, and collects multiple types of behavioral responses.

### Experimental Structure

- **Sessions**: Up to 5 sessions per participant
- **Blocks per session**: 8 blocks (2 blocks for practice/pre-sessions)
- **Trials per block**: 5 trials
- **Cue presentations per trial**: 5 cues
- **Images per block**: 3 images (2 common stimuli + 1 novel stimulus)

### Task Components

1. **Stimulus Presentation**: Visual cues with three novelty levels
2. **Reward Feedback**: Positive (+10) or negative (-50) outcomes
3. **Hedonic Ratings**: Linear scale from -5 (unpleasant) to +5 (pleasant)
4. **Causal Attribution Ratings**: Linear scale from 0 (not at all) to 10 (very likely)
5. **Bayesian Betting Task**: Triangular probability distribution over three images

## Files

### Main Scripts

#### `causal_inference_exp.m`
The main experimental script that runs the complete experiment session. This script:
- Manages multi-session protocols
- Presents stimuli and rewards according to schedule
- Collects all participant responses
- Saves comprehensive data files
- Handles image tracking across sessions

**Usage:**
```matlab
causal_inference_exp(EXP_NAME, index_num, session_opt)
```

**Parameters:**
- `EXP_NAME`: String identifier (e.g., 'test', 'subj001')
- `index_num`: Session number (1-5)
- `session_opt`: 'pre' for practice session, 'main' for main experiment

**Example:**
```matlab
% Run first practice session
causal_inference_exp('test', 1, 'pre');

% Run first main session
causal_inference_exp('test', 1, 'main');

% Run second main session
causal_inference_exp('test', 2, 'main');
```

#### `event_random.m`
Generates a completely randomized experimental schedule for the first session. Creates stimulus presentation sequences and reward assignments independently of any previous data.

**Usage:**
```matlab
event_random(subject, block_num)
```

**What it does:**
- Creates 8 blocks × 5 trials schedule
- Randomly assigns stimuli (types 1, 2, 3) to cue positions
- Randomly assigns rewards (+10 or -50) to trials
- Saves schedule to `results_save/{subject}_task_schedule_session{block_num}.mat`

**Called automatically by:** `causal_inference_exp.m` when `index_num == 1`

#### `given_event_made.m`
Generates experimental schedule for subsequent sessions based on previous session results. Enables progressive/adaptive experimental designs where later sessions are influenced by participant responses.

**Usage:**
```matlab
given_event_made(subject, BLOCK_NUM)
```

**What it does:**
- Loads previous session data from `./data/result_img_{subject}{BLOCK_NUM-1}.mat`
- Maps previous session responses to novelty levels
- Creates new schedule maintaining experimental structure
- Saves schedule to `results_save/{subject}_task_schedule_session{BLOCK_NUM}.mat`

**Called automatically by:** `causal_inference_exp.m` when `index_num > 1`

### Supporting Files

- **`seed.vol1.egg`** and **`seed.vol2.egg`**: Archive files containing seed images (extract to `./seed/` directory)
- **`Instruction.pdf`**: Detailed experimental instructions and protocol documentation

## Installation and Setup

### Requirements

1. **MATLAB** (R2014b or later recommended)
2. **Psychtoolbox-3** ([Download here](http://psychtoolbox.org/))
3. **Seed Images**: Extract `seed.vol1.egg` and `seed.vol2.egg` to create `./seed/` directory

### Directory Structure

```
Causal inference task/
├── causal_inference_exp.m      # Main experiment script
├── event_random.m              # Random schedule generator
├── given_event_made.m          # Progressive schedule generator
├── seed/                       # Directory for seed images
│   ├── 001.png, 002.png, ...   # Stimulus images
│   ├── msg_p10pt.png, ...      # Reward message images
│   ├── rating1.jpg, rating2.jpg # Rating scale images
│   └── ...
├── results_save/               # Created automatically - stores all data
└── data/                       # Created automatically - stores intermediate results
```

### Setup Steps

1. **Install Psychtoolbox**:
   ```matlab
   % Follow instructions at http://psychtoolbox.org/download
   ```

2. **Extract seed images**:
   - Extract `seed.vol1.egg` and `seed.vol2.egg` to create `./seed/` directory
   - Ensure images are named `001.png`, `002.png`, etc.

3. **Verify paths**:
   - Ensure `seed/` directory is in the same folder as the scripts
   - Scripts will create `results_save/` and `data/` directories automatically

## Running the Experiment

### Basic Workflow

1. **Start MATLAB** and navigate to the `Causal inference task` directory

2. **Run practice session** (optional but recommended):
   ```matlab
   causal_inference_exp('test', 1, 'pre');
   ```

3. **Run main experimental sessions**:
   ```matlab
   % Session 1
   causal_inference_exp('test', 1, 'main');
   
   % Session 2 (uses data from session 1)
   causal_inference_exp('test', 2, 'main');
   
   % Continue for sessions 3-5 as needed
   ```

### Session Types

- **'pre' (Practice Session)**:
  - 2 blocks instead of 8
  - Longer time limits for ratings (10s vs 7s)
  - Faster timing (4x speed) for practice
  - Helps participants learn the task

- **'main' (Main Session)**:
  - 8 blocks
  - Normal timing parameters
  - Full experimental protocol

### Keyboard Controls

During the experiment, participants use:
- **Left/Right Arrow Keys**: Navigate rating scales
- **'Y' Key**: Confirm/submit rating
- **'N' Key**: Reset (triangular task only)
- **'K' Key**: Skip to rating phase (if conditions met)
- **'Q' Key**: Quit/abort experiment

## Output Files

### Main Data File
**Location:** `results_save/{EXP_NAME}_main_{index_num}.mat` (or `_pre_{index_num}.mat`)

**Contains:**
- `HIST_bet_score_table_type1`: Triangular betting scores (barycentric coordinates)
- `HIST_bet_money_table_type1`: Monetary bets for each image combination
- `HIST_bet_score_table_type2`: Linear rating scores (hedonic and causal)
- `HIST_schedule`: Complete stimulus presentation schedule
- `HIST_reward`: Reward values for each trial
- `HIST_event_info`: Detailed event timing and type logging
- `Tot_img_info`: Image usage tracking across sessions
- All experimental parameters and settings

### Image Usage Tracking
**Location:** `results_save/{EXP_NAME}_image_usage_info.mat`

**Contains:**
- `Tot_img_info`: Matrix tracking which images were used in which sessions/blocks
- `HIST_event_info`: Event timing data for all sessions
- Prevents image reuse across sessions

### Task Schedule
**Location:** `results_save/{EXP_NAME}_task_schedule_session{index_num}.mat`

**Contains:**
- `HIST_schedule`: 1×8 cell array, each cell is 5×5 matrix (trials × cues)
- `HIST_reward`: 8×5 matrix (blocks × trials)

## Data Structure

### Event Logging

The `HIST_event_info` variable logs all experimental events with timestamps:

**Structure:** `[block#, trial#, trial_s#, session_time, block_time, event_type]`

**Event Type Codes:**
- `0.5`: Block start message displayed
- `-0.5`: Block start message removed
- `1, 2, 3`: Cue novelty level 1, 2, or 3 displayed
- `4`: Reward delivered
- `5`: Hedonic rating screen displayed
- `6`: Hedonic rating submitted
- `7`: Causal rating screen displayed
- `8`: Causal rating submitted
- `9`: Triangular betting screen displayed
- `10`: Triangular betting submitted
- `20`: Brief blank screen
- `-99`: Response timeout (failed to respond in time)
- Negative values: Display off events (e.g., `-1` = level 1 cue removed)

### Rating Data

**Type 2 Ratings (Linear Scales):**
- `HIST_bet_score_table_type2{block, 1}`: Hedonic ratings (-5 to +5)
- `HIST_bet_score_table_type2{block, 2}`: Causal attribution ratings (0 to 10)

**Type 1 Ratings (Triangular Betting):**
- `HIST_bet_score_table_type1{block, combination}`: Barycentric coordinates [p1, p2, p3] (sums to 1)
- `HIST_bet_money_table_type1{block, combination}`: Monetary bets for each image

## Experimental Parameters

### Key Parameters (in `causal_inference_exp.m`)

**Timing:**
- Cue display: 0.7 seconds
- Cue interval: 0.2-0.3 seconds (jittered)
- Reward display: 2 seconds
- Rating time limits: 7 seconds (main), 10 seconds (pre)

**Rewards:**
- Positive: +10 points
- Negative: -50 points
- Probability: 25% negative, 75% positive

**Stimulus Novelty:**
- Level 1: Common (appears 16 times in pool)
- Level 2: Medium (appears 8 times in pool)
- Level 3: Novel (appears once, shown once per block)

**Display:**
- Screen resolution: 1920×1080
- Cue image size: 256×256 pixels (scaled 1.5x for display)

## Troubleshooting

### Common Issues

1. **"Previous session file missing"**
   - Ensure you've run previous sessions in order
   - Check that `results_save/` directory contains previous session files

2. **"Session file already exists"**
   - Previous session with same name/number already completed
   - Use different `EXP_NAME` or `index_num`, or delete existing file

3. **Psychtoolbox errors**
   - Verify Psychtoolbox installation: `PsychtoolboxVersion`
   - Check screen synchronization settings
   - Ensure graphics drivers are up to date

4. **Missing seed images**
   - Extract `seed.vol1.egg` and `seed.vol2.egg`
   - Verify `./seed/` directory exists with images named `001.png`, `002.png`, etc.

5. **Image loading errors**
   - Check that seed images are in PNG format
   - Verify image file names match expected format (`%03d.png`)

### Debugging Tips

- Set `ffw_speed=4` for faster testing
- Use `'pre'` sessions to test without full 8-block protocol
- Check `HIST_event_info` for detailed timing of all events
- Monitor `Tot_img_info` to track image usage

## Advanced Usage

### Customizing Parameters

Key parameters can be modified in `causal_inference_exp.m`:

- **Block/Trial Structure**: Modify `Tot_block`, `Tot_trial`, `Tot_trial_s`
- **Timing**: Adjust `sec_stim_display`, `sec_limit_Q_rating`, etc.
- **Rewards**: Change `reward_mag` and `reward_prob`
- **Display**: Modify `IMAGE_SIZE`, `disp_scale`, colors, etc.

### Multi-Session Protocols

The script automatically:
- Tracks image usage across sessions (prevents reuse)
- Generates progressive schedules based on previous responses
- Maintains session continuity through file checking

### Integration with Other Tools

The saved data files can be:
- Loaded in MATLAB for analysis
- Exported to other formats (CSV, etc.) for statistical analysis
- Used with the Python models in `../Causal learning model/` and `../causal uncertainty model/`

## License

This project is licensed under the MIT License.

## Contact and Support

For questions about:
- **Experimental protocol**: See `Instruction.pdf`
- **Code functionality**: Review detailed comments in each `.m` file
- **Original paper**: Lee et al. (2015) PLoS Biology

## Notes

- The experiment is designed for use in fMRI/MRI scanners (includes scanner stabilization time)
- All timing can be adjusted via `ffw_speed` parameter for behavioral-only studies
- Image files are large (seed.vol files are ~37MB total) - ensure sufficient disk space
- The script creates detailed event logs suitable for fMRI analysis (timing precision)
