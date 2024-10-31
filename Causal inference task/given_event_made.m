% given_event_made.m
% Uses results from the previous session to generate an event schedule for the current session.
% Saves data to HIST_reward and HIST_schedule.

function given_event_made(subject, BLOCK_NUM)
    % Loads previous session results to generate an event schedule for the current session.
    % The function saves HIST_reward and HIST_schedule to a .mat file for the current session.
    
    % Load previous session result image data
    previous_result_path = fullfile('./data', sprintf('result_img_%s%d.mat', subject, BLOCK_NUM - 1));
% Load image result data from the previous block
    load(previous_result_path, 'result_img'); % Load result_img for the previous block
    
    % Initialize reward and schedule storage
    num_events = 8;                  % Number of events/blocks
    num_trials = 5;                  % Number of trials per event
% Initialize HIST_reward and HIST_schedule to store results
    HIST_reward = zeros(num_events, num_trials);
    HIST_schedule = cell(1, num_events);
    
    %% Map Results to Experimental Index
    % For each generated image (event), map result_img data to experiment schedule.
    for event = 1:num_events
        event_data = squeeze(result_img(event, :, :)); % Extract data for the event
        reward_part = event_data(:, 6);                % Column 6 contains reward information
        stimulus_part = event_data(:, 1:5);            % Columns 1-5 contain stimulus info
        
        % Sort stimulus to identify novelty levels
        [row_indices, col_indices] = ndgrid(1:size(stimulus_part, 1), 1:size(stimulus_part, 2));
% Sort the stimuli to assess novelty levels
        [sorted_values, sorted_idx] = sort(stimulus_part(:)); % Sort to categorize novelty
        
        % Identify novelty levels by index
        sorted_indices = [row_indices(sorted_idx), col_indices(sorted_idx)];
        high_novelty_idx = sorted_indices(end, :);             % Index of highest novelty (3)
        medium_novelty_idx = sorted_indices(end-8:end-1, :);   % Indices of medium novelty (2)
        low_novelty_idx = sorted_indices(1:end-9, :);          % Indices of lowest novelty (1)
        
        % Initialize stimulus schedule for this event
        stimulus_schedule = zeros(num_trials, num_trials);
        for i = 1:size(high_novelty_idx, 1)
            stimulus_schedule(high_novelty_idx(i, 1), high_novelty_idx(i, 2)) = 3;
        end
        for i = 1:size(medium_novelty_idx, 1)
            stimulus_schedule(medium_novelty_idx(i, 1), medium_novelty_idx(i, 2)) = 2;
        end
        for i = 1:size(low_novelty_idx, 1)
            stimulus_schedule(low_novelty_idx(i, 1), low_novelty_idx(i, 2)) = 1;
        end
        
        % Determine rewards, setting -50 for the maximum reward position
        [~, max_reward_idx] = max(reward_part);
        reward_schedule = ones(num_trials, 1) * 10; % Initialize all rewards to 10
        reward_schedule(max_reward_idx) = -50;       % Set maximum reward index to -50
        HIST_reward(event, :) = reward_schedule;     % Save reward schedule
        
        % Store the eventâs stimulus schedule
        HIST_schedule{1, event} = stimulus_schedule;
    end
    
    % Define the save path and save the schedule and rewards
    save_path = fullfile(pwd, 'results_save', sprintf('%s_task_schedule_session%d.mat', subject, BLOCK_NUM));
    save(save_path, 'HIST_reward', 'HIST_schedule');
end
