% given_event_made.m
% ============================================================================
% Generates event schedule for current session based on results from the
% previous session. This function enables progressive/adaptive experimental
% designs where later sessions are influenced by participant responses in
% earlier sessions.
%
% This function is called automatically by causal_inference_exp.m when
% index_num > 1 (subsequent sessions after the first).
%
% Inputs:
%   subject    - String identifier for the subject/experiment (e.g., 'test')
%   BLOCK_NUM  - Integer current session number (must be > 1)
%
% Outputs:
%   Saves to file: results_save/{subject}_task_schedule_session{BLOCK_NUM}.mat
%     - HIST_reward:  8x5 matrix of reward values (8 blocks x 5 trials)
%     - HIST_schedule: 1x8 cell array, each cell contains 5x5 matrix
%                      (5 trials x 5 cue presentations per trial)
%
% Dependencies:
%   Requires previous session data file:
%     ./data/result_img_{subject}{BLOCK_NUM-1}.mat
%   This file should contain 'result_img' variable with dimensions:
%     [num_events, num_trials, num_columns]
%   Where columns are: [stimulus_data (cols 1-5), reward_data (col 6)]
%
% Algorithm:
%   1. Loads result_img from previous session
%   2. For each block, extracts stimulus and reward information
%   3. Maps stimulus values to novelty levels (1=low, 2=medium, 3=high)
%   4. Assigns rewards based on previous session patterns
%   5. Creates schedule matching experimental structure
%
% Author: Based on Lee et al. (2015) one-shot learning paradigm
% ============================================================================

function given_event_made(subject, BLOCK_NUM)
    
    %% Load Previous Session Data
    % Construct path to previous session's result file
    previous_result_path = fullfile('./data', ...
                                   sprintf('result_img_%s%d.mat', subject, BLOCK_NUM - 1));
    
    % Check if previous session file exists
    if ~exist(previous_result_path, 'file')
        error('Previous session file not found: %s\nMake sure session %d was completed.', ...
              previous_result_path, BLOCK_NUM - 1);
    end
    
    % Load the result image data from previous session
    % result_img dimensions: [num_events, num_trials, num_columns]
    % Columns: [stimulus_data (1-5), reward_data (6), ...]
    load(previous_result_path, 'result_img');
    
    %% Initialize Storage Arrays
    % Define experimental structure (must match causal_inference_exp.m)
    num_events = 8;                  % Number of blocks/events
    num_trials = 5;                  % Number of trials per block
    num_trial_s = 5;                 % Number of cue presentations per trial
    
    % Preallocate storage for new schedule
    HIST_reward = zeros(num_events, num_trials);  % Reward history: 8x5
    HIST_schedule = cell(1, num_events);          % Schedule: 1x8 cell array
    
    %% Process Each Block/Event
    % Loop through each block to create schedule based on previous session
    for event = 1:num_events
        
        % Extract data for this specific block/event
        % squeeze removes singleton dimensions: [num_trials, num_columns]
        event_data = squeeze(result_img(event, :, :));
        
        % Extract reward information (column 6)
        reward_part = event_data(:, 6);
        
        % Extract stimulus information (columns 1-5: one per cue presentation)
        stimulus_part = event_data(:, 1:num_trial_s);
        
        %% Map Stimuli to Novelty Levels
        % Create coordinate grid for stimulus positions
        [row_indices, col_indices] = ndgrid(1:size(stimulus_part, 1), ...
                                            1:size(stimulus_part, 2));
        
        % Flatten and sort stimulus values to identify novelty levels
        % Lower values = lower novelty, higher values = higher novelty
        [sorted_values, sorted_idx] = sort(stimulus_part(:));
        
        % Map sorted indices back to (row, col) coordinates
        sorted_indices = [row_indices(sorted_idx), col_indices(sorted_idx)];
        
        % Categorize stimuli by novelty level based on sorted values:
        % - Highest value (last in sorted) = novelty level 3 (most novel)
        % - Next 8 values = novelty level 2 (medium novelty)
        % - Remaining values = novelty level 1 (low novelty/common)
        high_novelty_idx = sorted_indices(end, :);              % Single highest (level 3)
        medium_novelty_idx = sorted_indices(end-8:end-1, :);   % Next 8 (level 2)
        low_novelty_idx = sorted_indices(1:end-9, :);           % Remaining (level 1)
        
        %% Create Stimulus Schedule for This Block
        % Initialize 5x5 schedule matrix (5 trials x 5 cue presentations)
        stimulus_schedule = zeros(num_trials, num_trial_s);
        
        % Assign novelty level 3 (most novel) to highest value position
        for i = 1:size(high_novelty_idx, 1)
            stimulus_schedule(high_novelty_idx(i, 1), high_novelty_idx(i, 2)) = 3;
        end
        
        % Assign novelty level 2 (medium) to medium value positions
        for i = 1:size(medium_novelty_idx, 1)
            stimulus_schedule(medium_novelty_idx(i, 1), medium_novelty_idx(i, 2)) = 2;
        end
        
        % Assign novelty level 1 (common) to low value positions
        for i = 1:size(low_novelty_idx, 1)
            stimulus_schedule(low_novelty_idx(i, 1), low_novelty_idx(i, 2)) = 1;
        end
        
        %% Create Reward Schedule for This Block
        % Find the position with maximum reward value from previous session
        [~, max_reward_idx] = max(reward_part);
        
        % Initialize all rewards to positive (10 points)
        reward_schedule = ones(num_trials, 1) * 10;
        
        % Set the maximum reward position to negative (-50 points)
        % This maintains the experimental structure where one trial per block
        % has a negative outcome
        reward_schedule(max_reward_idx) = -50;
        
        % Store reward schedule for this block
        HIST_reward(event, :) = reward_schedule;
        
        % Store stimulus schedule for this block
        HIST_schedule{1, event} = stimulus_schedule;
    end
    
    %% Save Schedule to File
    % Create save directory path
    save_path = fullfile(pwd, 'results_save', ...
                        sprintf('%s_task_schedule_session%d.mat', subject, BLOCK_NUM));
    
    % Ensure results_save directory exists
    if ~exist(fullfile(pwd, 'results_save'), 'dir')
        mkdir(fullfile(pwd, 'results_save'));
    end
    
    % Save the generated schedule and reward matrices
    save(save_path, 'HIST_reward', 'HIST_schedule');
    
    fprintf('Progressive schedule generated from session %d and saved to: %s\n', ...
            BLOCK_NUM - 1, save_path);
end
