
# Causal Inference Experiment Simulation

This repository contains MATLAB scripts used for simulating a causal inference experiment. The scripts generate randomized stimuli and reward sequences for experiment sessions and are designed to interact with Psychtoolbox.

## Files

1. **causal_inference_exp.m**: Simulates the main experimental session by presenting stimuli and recording participant responses in a visual-causal task.
2. **event_random.m**: Generates random sequences of stimuli and rewards for an experimental session. Saves results as `HIST_reward` and `HIST_schedule` matrices.
3. **given_event_made.m**: Uses results from a previous session to influence the current session's stimuli and rewards. Useful for sessions where progressive data influence is desired.

## Usage

Each script is meant to be run in a MATLAB environment with Psychtoolbox installed.

- **To run `event_random.m` or `given_event_made.m`**: Call the function with `subject` and `block_num` or `BLOCK_NUM` parameters.

## Dependencies

- [Psychtoolbox-3](http://psychtoolbox.org/) for MATLAB (required for displaying visual stimuli)

## License

This project is licensed under the MIT License.
