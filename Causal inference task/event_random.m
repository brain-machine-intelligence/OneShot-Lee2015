% event_random.m
% ============================================================================
% Generates random stimulus and reward sequences for the first experimental
% session. This function creates a completely randomized schedule that is
% independent of any previous session data.
%
% This function is called automatically by causal_inference_exp.m when
% index_num == 1 (first session).
%
% Inputs:
%   subject    - String identifier for the subject/experiment (e.g., 'test')
%   block_num  - Integer session number (should be 1 for first session)
%
% Outputs:
%   Saves to file: results_save/{subject}_task_schedule_session{block_num}.mat
%     - HIST_reward:  8x5 matrix of reward values (8 blocks x 5 trials)
%     - HIST_schedule: 1x8 cell array, each cell contains 5x5 matrix
%                      (5 trials x 5 cue presentations per trial)
%
% Stimulus Types:
%   - Type 1: Common stimulus (appears 16 times in pool)
%   - Type 2: Medium novelty stimulus (appears 8 times in pool)
%   - Type 3: Novel stimulus (appears 1 time in pool, shown only once per block)
%
% Reward Values:
%   - Positive: 10 points (appears 4 times)
%   - Negative: -50 points (appears 1 time)
%
% Author: Based on Lee et al. (2015) one-shot learning paradigm
% ============================================================================

function event_random(subject, block_num)
    
    %% Experimental Parameters
    % Define the structure of the experimental session
    num_events = 8;    % Number of blocks/events in the session
    num_trials = 5;    % Number of trials in each block
    num_trial_s = 5;   % Number of cue presentations per trial (5x5 grid)
    
    %% Initialize Storage Arrays
    % Preallocate matrices for efficient memory usage
    HIST_reward = zeros(num_events, num_trials);  % Reward history: 8 blocks x 5 trials
    HIST_schedule = cell(1, num_events);          % Schedule: 1x8 cell array
    
    %% Define Stimulus Pool
    % Create a pool of stimuli with different novelty levels
    % The ratio [16, 8, 1] matches the experimental design where:
    % - Stimulus type 1 (common): 16 occurrences
    % - Stimulus type 2 (medium): 8 occurrences  
    % - Stimulus type 3 (novel): 1 occurrence (will appear once per block)
    stimuli = [repmat(1, 1, 16), repmat(2, 1, 8), 3];
    
    %% Define Reward Pool
    % Create a pool of rewards for each trial
    % Positive rewards (10 points) appear 4 times, negative (-50) appears once
    % This creates a 4:1 ratio of positive to negative outcomes
    rewards = [10, 10, 10, 10, -50];
    
    %% Generate Random Schedule for Each Block
    % Loop through each block to create independent randomized schedules
    for event = 1:num_events
        
        % Randomly permute indices to shuffle stimuli and rewards
        stim_indices = randperm(numel(stimuli));   % Random order for stimuli
        reward_indices = randperm(numel(rewards));  % Random order for rewards
        
        % Create shuffled stimulus sequence
        stim_shuffled = stimuli(stim_indices);
        
        % Reshape into trial structure: 5 trials x 5 cue presentations
        % Takes first 25 stimuli (5*5) from shuffled pool
        event_schedule = reshape(stim_shuffled(1:num_trials * num_trial_s), ...
                                  [num_trials, num_trial_s]);
        
        % Assign shuffled rewards to trials in this block
        % Each trial gets one reward value from the shuffled pool
        HIST_reward(event, :) = rewards(reward_indices);
        
        % Store the complete schedule for this block
        HIST_schedule{1, event} = event_schedule;
    end
    
    %% Save Schedule to File
    % Create save directory path
    save_path = fullfile(pwd, 'results_save', ...
                        sprintf('%s_task_schedule_session%d.mat', subject, block_num));
    
    % Ensure results_save directory exists
    if ~exist(fullfile(pwd, 'results_save'), 'dir')
        mkdir(fullfile(pwd, 'results_save'));
    end
    
    % Save the generated schedule and reward matrices
    save(save_path, 'HIST_reward', 'HIST_schedule');
    
    fprintf('Random schedule generated and saved to: %s\n', save_path);
end
