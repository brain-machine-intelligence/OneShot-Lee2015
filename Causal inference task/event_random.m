% event_random.m
% Generates random stimulus and reward sequences for a block of experiment trials.
% Saves results as HIST_reward and HIST_schedule matrices.

function event_random(subject, block_num)
    % Generates random stimulus and reward sequences for the experiment session.
    % The function saves the generated HIST_reward and HIST_schedule to a .mat file.
    
    % Initialize reward and stimulus sequences
% Define the number of events and trials per event
    num_events = 8;    % Number of events/blocks in session
    num_trials = 5;    % Number of trials in each block
% Define the number of events and trials per event
    HIST_reward = zeros(num_events, num_trials); % Preallocate reward history
% Define the number of events and trials per event
    HIST_schedule = cell(1, num_events);         % Preallocate schedule cell array
    
    % Define stimulus and reward sets
% Set up the possible stimuli for trials
    stimuli = [repmat(1, 1, 16), repmat(2, 1, 8), 3]; % Stimulus types 1, 2, and 3
    rewards = [10, 10, 10, 10, -50];                % Reward values for trials
    
    % Loop over each event (block) to create randomized schedules
% Define the number of events and trials per event
    for event = 1:num_events
        % Randomize stimulus and reward sequences for this event
        stim_indices = randperm(numel(stimuli));
        reward_indices = randperm(numel(rewards));
        
        % Shuffle and reshape stimuli for the event
        stim_shuffled = stimuli(stim_indices);
        event_schedule = reshape(stim_shuffled(1:num_trials^2), [num_trials, num_trials]);
        
        % Shuffle rewards and assign to the event
        HIST_reward(event, :) = rewards(reward_indices);
        
        % Store schedule and reward for this event
        HIST_schedule{1, event} = event_schedule;
    end
    
    % Define file path for saving session schedule
    save_path = fullfile(pwd, 'results_save', sprintf('%s_task_schedule_session%d.mat', subject, block_num));
    
    % Save HIST_reward and HIST_schedule
% Save results to the designated file path
    save(save_path, 'HIST_reward', 'HIST_schedule');
end
