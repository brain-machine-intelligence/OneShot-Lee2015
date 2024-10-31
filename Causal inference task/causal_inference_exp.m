% causal_inference_exp.m
% This script is designed to simulate a causal inference experiment using Psychtoolbox.
% It presents visual stimuli, records participant responses, and saves the data for each session.
% The experiment can operate in different modes ('pre' or 'main') and is suitable for multiple sessions.

function [output_info]=causal_inference_exp(EXP_NAME,index_num,session_opt)
% Parameters:
% EXP_NAME : string - Identifier for the experiment name, e.g., 'test'.
% index_num : integer - Current session number, used to handle session-specific actions.
% session_opt : string - Sesscausal_inference_expion type, either 'pre' for preliminary or 'main' for primary experiment sessions.

% Session initialization based on session index number
if index_num==1
    event_random(EXP_NAME, index_num); % Run random event setup for the first session
else
    given_event_made(EXP_NAME, index_num); % Load preset events for subsequent sessions
end

% Disable sync tests in Psychtoolbox for debugging
Screen('Preference', 'SkipSyncTests', 1);

% Validate the session option input, raising an error for invalid values
if strcmp(session_opt, 'pre') || strcmp(session_opt, 'main')
    okay_to_start = 1; % Proceed if session type is valid
else
    error('ERROR: Invalid session_opt. Use "pre" or "main" only.');
end

% Set up paths for seed files and saving results
seed_path = fullfile(pwd, 'seed'); % Path to load experimental seeds
save_path = fullfile(pwd, 'results_save'); % Directory for saving results

% Check for file conflicts and ensure previous session file exists for continuity
file_chk_name = fullfile(save_path, sprintf('%s_main_%d.mat', EXP_NAME, index_num));
if exist(file_chk_name, 'file')
    disp('ERROR: Session file already exists. Try another session name or number.');
    return;
elseif index_num > 1
    previous_file = fullfile(save_path, sprintf('%s_main_%d.mat', EXP_NAME, index_num - 1));
    if ~exist(previous_file, 'file')
        disp('ERROR: Previous session file missing. Check previous session number.');
        return;
    end
elseif index_num > 5
    disp('ERROR: Maximum session limit of 5 exceeded.');
    return;
end

% Core options: resolution, image size, and display parameters
KEEP_PICTURE_OPT=1; % keepting picture + bonus session
SCREEN_RESOLUTION=[1920 1080];%[1920 1200];
IMAGE_SIZE=[256 256]; %width,height - for cue presentation
IMAGE_SIZE_Q_outcome = [1377 121];
IMAGE_SIZE_Q_hedonic = [839 111];
rating_size = [1514 169];
disp_scale=1.5; % for cue presentation
disp_scale_causalrating=1.0; % cues size for hedonic and causal attribution ratings
disp_scale_keepingpicture=0.55; % cue size for ratings for keeping pictures
size_triangle=110; % the size of triangle for keeping pictures
IMAGE_type1 = [1378 213];

% Timing and session-specific parameters
Tot_session=1; % # of total sessions (fixed because this runs for each session)
Tot_block=8; % # of total blocks (must be EVEN number)
if(strcmp(session_opt,'pre')==1) % pre-session
    Tot_block=2;
end
Tot_trial=5; % # of trials in each block
Tot_trial_s=5; % # of cue presentation in one trial [NOTE] Tot_trial*Tot_trial_s must be *even* number
tot_num_img_per_block=3;

ffw_speed=1; %1 % fast forward speed, 1: normal speed
if(strcmp(session_opt,'pre')==1) % pre-session
    ffw_speed=4; %4
end

% Set timing values for each component in seconds, adjusted for fast-forward speed
sec_scanner_ready=5/ffw_speed; % sec for scanner stabilization
sec_block_ready=0.5/ffw_speed; % sec for block ready signal
sec_stim_display=0.7/ffw_speed;
sec_stim_interval=[0.2 0.3]/(ffw_speed);%1.5; %(sec)
sec_trial_interval=[1 2]/(ffw_speed);%1.5; %(sec)
sec_limit_Q_rating=[7 7]; % time limit of [hedonic_rating causal_rating] (sec)
sec_limit_Bayesian_rating=8; % time limit of Bayesian triangular rating(sec)
sec_jittered_blank_page=0.15/(ffw_speed); % (sec)
if(strcmp(session_opt,'pre')==1) % pre-session
    sec_limit_Q_rating=[10 10]; % time limit of [hedonic_rating causal_rating] (sec)
    sec_limit_Bayesian_rating=20; % time limit of Bayesian triangular rating(sec)
end

earliest_novel_stimulus_show_in_block=0.6; % 0: any trial in block, 1: last?
earliest_reward_stimulus_show_in_block=0.6; % 0: any trial in block, 1: last?. must be in [reward_prob] range.
sec_reward_display=2/ffw_speed;
% how_early_novel_stimulus_show_in_trial=1; % 0: shown at 1-st trial, 1: any trial in block.
latest_novel_stimulus_show_in_trial_s=0.6; % 0: shown at 1-st trial_s, 1: any trial_s in trial.

% reward [neg rwd, pos rwd]
reward_mag=[[-50 10];[50 -10]]; % col1: novel outcome, col2: non-novel outcome, # of rows = # of outcome conditions
reward_prob=[0.25 0.75]; % must be 2nd col>1st colyy

% money
money_given=1; % point
% novelty-level for the non-novel stimuli. a novel stimulus will be shown only one time.
img_present_ratio=[2 1]; % novelty level :: size = tot_num_img_per_block-1.
%% options: display
% questions
Q{1,1}.color=[0, 0, 255, 255];
Q{1,1}.cursor_input=[-5:1:5];
Q{1,2}.text2='(Not at all)';
Q{1,2}.text3='(Very likely)';
Q{1,2}.text4='(Don''t know)';
Q{1,2}.color=[255, 0, 0, 255];
Q{1,2}.color_1=[0, 0, 255, 255];
Q{1,2}.cursor_input=[0:1:10];

% text size
text_size_default=20; % font size (don't change)
text_size_reward=400; % height in pixel

% background color
BackgroundColor_block_intro=[130,130,130,150]; % gray
BackgroundColor_Cue_page=[210,210,210,150]; % light gray
BackgroundColor_Trial_ready_page=[210,210,210,150]; % light gray
BackgroundColor_Reward_page=[210,210,210,150]; % light gray
COLOR_FIXATION_MARK=[70,70,70,200]; % dark gray
% automatically determined
total_bet_trial=nchoosek(tot_num_img_per_block,3);
list_combination=randperm(tot_num_img_per_block);


% key code (laptop)
KEY_L=37;
KEY_R=39;
KEY_Y=89; %'y'
KEY_N=78; %'n'
KEY_Q=81; %'q'
KEY_K=KbName('k'); % k
KEY_T=84; % 't', 5 in desktop, 84 in laptop

% recording variables (for each block)
HIST_bet_score_table_type1=cell(Tot_block,size(list_combination,1));
HIST_bet_money_table_type1=cell(Tot_block,size(list_combination,1));
HIST_Barycentric_coord_table_type1=cell(Tot_block,size(list_combination,1));
HIST_bet_score_table_type2=cell(Tot_block,size(Q,2));

HIST_bet_score_table_type1_bonus=cell(1,size(list_combination,1));
HIST_bet_money_table_type1_bonus=cell(1,size(list_combination,1));
HIST_Barycentric_coord_table_type1_bonus=cell(1,size(list_combination,1));
HIST_bet_score_table_type2_bonus=cell(1,size(Q,2));

HIST_event_info0=[]; % row1=event time(in session), row2=event time(in block), row3=event type
HIST_event_info_Tag{1,1}='row1 - block#';    HIST_event_info_Tag{2,1}='row2 - trial#, 0 if outside of the trial';     HIST_event_info_Tag{3,1}='row3 - trial_s#, 0 if outside of the trial_s';
HIST_event_info_Tag{4,1}='row4 - event time in session';   HIST_event_info_Tag{5,1}='row5 - event time in block';
HIST_event_info_Tag{6,1}='row6 - 0.5: block start msg on, -0.5: block start msg off, 1: cue novelty level1, 2: cue novelty level2, 3: cue novelty level3, 4: reward delivery, 5: hedonic rating display, 6: hedonic rating answer, 7: causal rating display, 8: causal rating answer, 9: Bayesian rating display, 10: Bayesian rating answer, 20: a short blank page display, -99:fail to do ratings in time limit, (-) when display off';




%% Seed Image Loading and Initialization
% In the first session, initialize the image usage matrix, or load the matrix if it's a subsequent session.

if(index_num==1)
    Tot_num_img=120; %Tot_block*tot_num_img_per_block;
    Tot_img_info=[1:1:Tot_num_img];
    Tot_img_info=[Tot_img_info; zeros(8,Tot_num_img)]; % 2nd row: used index, 3rd row: session#, 4th row: block#, 5th row: novelty level (1:non-novel, 3:novel)
    Tot_img_info_Tag{1,1}='row1 - file #';    Tot_img_info_Tag{2,1}='row2 - is used';
    Tot_img_info_Tag{3,1}='row3 - session #';    Tot_img_info_Tag{4,1}='row4 - block #';
    Tot_img_info_Tag{5,1}='row5 - novelty level (1:non-novel, 3:novel)';        Tot_img_info_Tag{6,1}='row6 - ratings (hedonic)';
    Tot_img_info_Tag{7,1}='row7 - rating (bad outcome causality)';     Tot_img_info_Tag{8,1}='row8 - ratings (bayesian for, the bonus round)';
    Tot_img_info_Tag{9,1}='row9 - case 0: nov-pos, 1:nov-neg';
    if(strcmp(session_opt,'pre')==1) % pre-session
        Tot_num_img=3* Tot_block;
        Tot_img_info=[1:1:Tot_num_img];
        Tot_img_info=[Tot_img_info; zeros(8,Tot_num_img)]; % 2nd row: used index, 3rd row: session#, 4th row: block#, 5th row: novelty level (1:non-novel, 3:novel)
    end
else % load image usage matrix to update
    file_imgind_ld_name=[EXP_NAME '_image_usage_info.mat'];
    file_name_ld=[save_path file_imgind_ld_name];
    load(file_name_ld);
end

img_set=cell(Tot_block,tot_num_img_per_block-1);
img0_set=cell(Tot_block,1);
img_set_bonus=cell(Tot_block,tot_num_img_per_block-1);
img0_set_bonus=cell(Tot_block,1);

HIST_current_pool=[];
for block=1:1:Tot_block

    % select all images to be used for each block (nonoverlap among blocks)
    [tmp current_img_pool_ind]=find(Tot_img_info(2,:)==0);
    current_pool=Tot_img_info(1,current_img_pool_ind);
    img_index_use_all=randsample(current_pool,tot_num_img_per_block,false);

    % update image usage matrix
    Tot_img_info(2,img_index_use_all)=1; % tagged as "used"
    Tot_img_info(3,img_index_use_all)=index_num; % record the current session#
    Tot_img_info(4,img_index_use_all)=block; % record the current block#
    Tot_img_info(5,img_index_use_all(1))=1; % record the novelty level
    Tot_img_info(5,img_index_use_all(2))=2; % record the novelty level
    Tot_img_info(5,img_index_use_all(3))=3; % record the novelty level

    % assign normal&novel stimulus
    img_index_use=img_index_use_all(1:end-1);
    img0_index_use=img_index_use_all(end);

    num_stim=length(img_index_use);
    for i=1:1:num_stim
        file_full_path=[seed_path '\' sprintf('%03d.png',img_index_use(i))];
        if(strcmp(session_opt,'pre')==1) % pre-session
            file_full_path=[seed_path '\' sprintf('%03d.png',img_index_use(i))];
        end
        img_set{block,i}=imread(file_full_path); % get a frame
    end
    num_stim0=length(img0_index_use);
    for i=1:1:num_stim0
        file_full_path=[seed_path '\' sprintf('%03d.png',img0_index_use(i))];
        if(strcmp(session_opt,'pre')==1) % pre-session
            file_full_path=[seed_path '\' sprintf('%03d.png',img0_index_use(i))];
        end
        % img0_set{block,i}=imresize(imread(file_full_path),[IMAGE_SIZE(2) IMAGE_SIZE(1)]);  % get a frame
        img0_set{block,i}=imread(file_full_path);
    end
    HIST_current_pool=[HIST_current_pool; [img_index_use img0_index_use]];

end

% reward message image read
img_msg_set=cell(size(reward_mag,2),size(reward_mag,1));
for mm=1:1:size(reward_mag,1)    % reward case
    for jj=1:1:size(reward_mag,2)   % reward set for each case
        if(reward_mag(mm,jj)>=0)
            file_full_path=[seed_path '\' sprintf('msg_p%02dpt.png',abs(reward_mag(mm,jj)))];
        else
            file_full_path=[seed_path '\' sprintf('msg_n%02dpt.png',abs(reward_mag(mm,jj)))];
        end
        img_msg_set{jj,mm}=imread(file_full_path);
    end
end
%% Load Question Images for Display
% Loading images for question prompts on causal and hedonic inference
question_set = cell(1,3);
question_set{1,1} = imread([seed_path '\tmp_good_infer.jpg']);
question_set{1,2} = imread([seed_path '\tmp_bad_infer.jpg']);
question_set{1,3} = imread([seed_path '\tmp_hedonic.jpg']);

%% Scheduling Setup
% Define frequency of image presentations per block
img_present_freq=img_present_ratio/sum(img_present_ratio);

ind_mat=[];     num_acc=0;
tot_num=Tot_trial*Tot_trial_s;

HIST_schedule=cell(1,Tot_block);
HIST_reward=zeros(Tot_block,Tot_trial);
HIST_reward_case=zeros(Tot_block,1);
for j=1:1:tot_num_img_per_block-2
    numm=round((tot_num-1)*img_present_freq(j));
    num_acc=num_acc+numm;
    ind_mat=[ind_mat j*ones(1,numm)];
end
ind_mat=[ind_mat (tot_num_img_per_block-1)*ones(1,tot_num-1-num_acc)]; % second last column
case_seed0=[ones(1,Tot_block/2) ones(1,Tot_block/2)]';
HIST_reward_case=case_seed0(randperm(Tot_block));
task_file_name = [pwd '\results_save\' EXP_NAME '_' sprintf('task_schedule_session%d.mat',index_num)];
load(task_file_name);

%% Display initialization
whichScreen = 0;
wPtr  = Screen('OpenWindow',whichScreen);
[screenWidth, screenHeight] = Screen('WindowSize', wPtr);

white = WhiteIndex(wPtr); % pixel value for white
black = BlackIndex(wPtr); % pixel value for black
gray = (white+black)/2;
inc = white-gray;
inc_0=white-black;
HIST_key_press_button=ones(1,Tot_trial);
HIST_key_press_sec=ones(1,Tot_trial);

%% Starting Message
% Display a message at the start of the experiment asking the participant if they are ready.
Screen('TextSize',wPtr, text_size_default);
DrawFormattedText(wPtr, 'Are you ready for the experiment?\n(Press any key to start)', 'center', 'center');
Screen('Flip', wPtr);
KbWait; % temporarily disabled for test APR 21

%% clock-start and then wait for another 5secs until the scanner stabilizes
session_clock_start = GetSecs;
WaitSecs(sec_scanner_ready);

%% Block Loop
img_set_all=cell(Tot_block,tot_num_img_per_block);
for block=1:1:Tot_block % Loop through each block in the session

    %% I. Stimulus Presentation Phase
    % Start of block message
    Screen('FillRect',wPtr,BackgroundColor_block_intro);
    str_block_intro=sprintf('*** Now block %d starts. ***',block);
    DrawFormattedText(wPtr, str_block_intro, 'center', 'center',[0, 0, 0, 255]);
    Screen(wPtr, 'Flip');
    block_clock_start = GetSecs;
    HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); 0.5]]; % event save
    WaitSecs(sec_block_ready);

    % add fixation mark and display off during the jittered interval
    DrawFormattedText(wPtr, '+', 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.
    Screen(wPtr, 'Flip');
    HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); -0.5]]; % event save
    sec_stim_interval0=rand*(max(sec_stim_interval)-min(sec_stim_interval))+min(sec_stim_interval);
    WaitSecs(sec_stim_interval0);

    img0_index_showed=zeros(1,length(img0_index_use));

    for trial=1:1:Tot_trial % each trial

        Screen('FillRect',wPtr,BackgroundColor_Trial_ready_page)

        flag_q_pressed = 0;

        for trial_s=1:1:Tot_trial_s % cue presentation

            % Determine which stimlus will be presented.
            selected=HIST_schedule{1,block}(trial,trial_s);
            if(selected==tot_num_img_per_block) % the most novel cue
                input_stim = Screen('MakeTexture', wPtr, img0_set{block,1});
                img0_index_showed(1)=1;
            else
                input_stim = Screen('MakeTexture', wPtr, img_set{block,selected});
            end

            % image display
            xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
            sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
            destrect=[xpos-sx/2,ypos-sy/2,xpos+sx/2,ypos+sy/2];
            Screen('FillRect',wPtr,BackgroundColor_Cue_page);
            Screen('DrawTexture', wPtr, input_stim,[],destrect);
            Screen(wPtr, 'Flip'); % display on
            HIST_event_info0=[HIST_event_info0 [block; trial; trial_s; (GetSecs-session_clock_start); (GetSecs-block_clock_start); selected]]; % event save
            WaitSecs(sec_stim_display);
            % add fixation mark and display off during the jittered interval
            DrawFormattedText(wPtr, '+', 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.
            Screen(wPtr, 'Flip');
            HIST_event_info0=[HIST_event_info0 [block; trial; trial_s; (GetSecs-session_clock_start); (GetSecs-block_clock_start); (-1)*selected]]; % event save
            sec_stim_interval0=rand*(max(sec_stim_interval)-min(sec_stim_interval))+min(sec_stim_interval);
            %WaitSecs(sec_stim_interval0);
            clock_time_limit_start=GetSecs;

            % Keyboard wait for input or timeout during cue presentation
            while 1
                [secs, keyCode] = KbPressWait([], clock_time_limit_start + sec_stim_interval0); % if no keyboard in time limit, then go ahead. if pressed earlier, then go ahead.
                [tmp tmp_key_code]=find(keyCode==1);
                if(tmp_key_code==KEY_K) % K pressed
                    if length(img_index_use) ==2 && sum(img0_index_showed)==1

                        flag_q_pressed = 1;
                        break;
                    end
                end

                if((GetSecs-clock_time_limit_start)> sec_stim_interval0)
                    break;
                end
            end

            if flag_q_pressed ==1
                break;
            end
        end

        if flag_q_pressed ==1
            break;
        end

        %% Reward Display Phase
        Screen('FillRect',wPtr,BackgroundColor_Reward_page);
        if(HIST_reward_case(block,1)==1) % non-novelO:10, novelO:-50
            if(HIST_reward(block,trial)<0) % novel outcome
                input_stim_msg = Screen('MakeTexture', wPtr, img_msg_set{1,HIST_reward_case(block,1)});
                sx_msg=size(img_msg_set{1,HIST_reward_case(block,1)},2);    sy_msg=size(img_msg_set{1,HIST_reward_case(block,1)},1);
            else % non-novel outcome
                input_stim_msg = Screen('MakeTexture', wPtr, img_msg_set{2,HIST_reward_case(block,1)});
                sx_msg=size(img_msg_set{2,HIST_reward_case(block,1)},2);    sy_msg=size(img_msg_set{2,HIST_reward_case(block,1)},1);
            end
        end
        if(HIST_reward_case(block,1)==2) % non-novelO:-10, novelO:50
            if(HIST_reward(block,trial)<0) % non-novel outcome
                input_stim_msg = Screen('MakeTexture', wPtr, img_msg_set{2,HIST_reward_case(block,1)});
                sx_msg=size(img_msg_set{1,HIST_reward_case(block,1)},2);    sy_msg=size(img_msg_set{1,HIST_reward_case(block,1)},1);
            else % novel outcome
                input_stim_msg = Screen('MakeTexture', wPtr, img_msg_set{1,HIST_reward_case(block,1)});
                sx_msg=size(img_msg_set{2,HIST_reward_case(block,1)},2);    sy_msg=size(img_msg_set{2,HIST_reward_case(block,1)},1);
            end
        end
        destrect=[xpos-sx_msg/2,ypos-sy_msg/2,xpos+sx_msg/2,ypos+sy_msg/2];
        Screen('DrawTexture', wPtr, input_stim_msg,[],destrect);
        Screen(wPtr, 'Flip');
        HIST_event_info0=[HIST_event_info0 [block; trial; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); 4]]; % event save
        clock_time_limit_start=GetSecs;

        while 1
            [secs, keyCode] = KbPressWait([], clock_time_limit_start + sec_reward_display); % if no keyboard in time limit, then go ahead. if pressed earlier, then go ahead.
            [tmp tmp_key_code]=find(keyCode==1);
            if(tmp_key_code==KEY_K) % L pressed
                if length(img_index_use) ==2 && sum(img0_index_showed)==1

                    flag_q_pressed = 1;
                    break;
                end
            end

            if((GetSecs-clock_time_limit_start)> sec_reward_display)
                break;
            end
        end

        if flag_q_pressed == 1
            break;
        end
        % add fixation mark and display off during the jittered interval
        DrawFormattedText(wPtr, '+', 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.
        Screen(wPtr, 'Flip');
        HIST_event_info0=[HIST_event_info0 [block; trial; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); -4]]; % event save
        sec_trial_interval0=rand*(max(sec_trial_interval)-min(sec_trial_interval))+min(sec_trial_interval);
        WaitSecs(sec_trial_interval0);

        if flag_q_pressed == 1
            break;
        end

    end

    % Collect all images used in the block
    for h=1:1:length(img_index_use)
        img_set_all{block,h}=img_set{block,h};
    end
    [tmp ind_new]=find(img0_index_showed==1);

    for h=1:1:sum(img0_index_showed)
        img_set_all{block,h+length(img_index_use)}=img0_set{block,ind_new(h)};
    end

    %% III. Rating - type2
    % This section manages the type 2 rating phase, displaying images and a rating scale to the participant.

    img_toshow_index=randperm(tot_num_img_per_block);
    xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
    length_bar=round(screenWidth*3/5);
    bar_pt_start=[round(screenWidth*1/5),screenHeight-200];
    bar_pt_end=[round(screenWidth*1/5)+length_bar,screenHeight-200];
    bet_score_table=[img_toshow_index; zeros(1,3)];
    bet_money_table=[img_toshow_index; zeros(1,3)];

    for q_ind=1:1:size(Q,2)

        ggg_ind=0;
        for m=1:1:tot_num_img_per_block
            ggg_ind=ggg_ind+1;

            Is_bet=0;
            bar_pt_cursor=(Q{1,q_ind}.cursor_input-min(Q{1,q_ind}.cursor_input))/(max(Q{1,q_ind}.cursor_input)-min(Q{1,q_ind}.cursor_input)); %[0,1]
            bar_pt_cursor=length_bar*bar_pt_cursor+bar_pt_start(1); %actual x-positions in display
            current_cursor_selection_ind=round(length(Q{1,q_ind}.cursor_input)/2);
            ind_ev=0;

            while(~Is_bet)

                ind_ev=ind_ev+1;

                %% Display Rating Questions
                % Customize display based on the question type and current context
                if q_ind==2
                    if (length(find(HIST_reward(block,:)>0))==1)
                        % Display positive reward outcome
                        sx = floor(IMAGE_SIZE_Q_outcome(1)*0.7);
                        sy = floor(IMAGE_SIZE_Q_outcome(2)*0.7);
                        sx1 = floor(rating_size(1)*0.7);
                        sy1 = floor(rating_size(2)*0.7);
                        disp_color=Q{1,2}.color_1;
                        input_stim = Screen('MakeTexture', wPtr, question_set{1,1});
                        rating = imread([seed_path 'rating2.jpg']);
                        input_stim2 = Screen('MakeTexture', wPtr, rating);
                        img_pt_center=[xpos, bar_pt_start(2)-sy/2-380];
                        img_pt_center2=[xpos, bar_pt_start(2)+sy1/2+50];
                        destrect=[img_pt_center(1)-sx/2, img_pt_center(2)-sy/2, img_pt_center(1)+sx/2, img_pt_center(2)+sy/2];
                        destrect2=[img_pt_center2(1)-sx1/2, img_pt_center2(2)-sy1/2, img_pt_center2(1)+sx1/2, img_pt_center2(2)+sy1/2];
                        Screen('DrawTexture',wPtr, input_stim, [], destrect);
                        Screen('DrawTexture',wPtr, input_stim2, [], destrect2);
                    else
                        % Display negative reward outcome
                        sx = floor(IMAGE_SIZE_Q_outcome(1)*0.7);
                        sy = floor(IMAGE_SIZE_Q_outcome(2)*0.7);
                        sx1 = floor(rating_size(1)*0.7);
                        sy1 = floor(rating_size(2)*0.7);
                        disp_color=Q{1,2}.color;
                        input_stim = Screen('MakeTexture', wPtr, question_set{1,2});
                        rating = imread([seed_path 'rating2.jpg']);
                        input_stim2 = Screen('MakeTexture', wPtr, rating);
                        img_pt_center=[xpos, bar_pt_start(2)-sy/2-380];
                        img_pt_center2=[xpos, bar_pt_start(2)+sy1/2+50];
                        destrect=[img_pt_center(1)-sx/2, img_pt_center(2)-sy/2, img_pt_center(1)+sx/2, img_pt_center(2)+sy/2];
                        destrect2=[img_pt_center2(1)-sx1/2, img_pt_center2(2)-sy1/2, img_pt_center2(1)+sx1/2, img_pt_center2(2)+sy1/2];
                        Screen('DrawTexture',wPtr, input_stim, [], destrect);
                        Screen('DrawTexture',wPtr, input_stim2, [], destrect2);
                    end
                elseif q_ind==1 % For hedonic-based questions
                    sx = floor(IMAGE_SIZE_Q_hedonic(1)*0.8);
                    sy = floor(IMAGE_SIZE_Q_hedonic(2)*0.8);
                    sx1 = floor(rating_size(1)*0.7);
                    sy1 = floor(rating_size(2)*0.7);
                    disp_color=Q{1,1}.color;
                    input_stim = Screen('MakeTexture', wPtr, question_set{1,3});
                    rating = imread([seed_path 'rating1.jpg']);
                    input_stim2 = Screen('MakeTexture', wPtr, rating);
                    img_pt_center=[xpos, bar_pt_start(2)-sy/2-380];
                    img_pt_center2=[xpos, bar_pt_start(2)+sy1/2+50];
                    destrect=[img_pt_center(1)-sx/2, img_pt_center(2)-sy/2, img_pt_center(1)+sx/2, img_pt_center(2)+sy/2];
                    destrect2=[img_pt_center2(1)-sx1/2, img_pt_center2(2)-sy1/2, img_pt_center2(1)+sx1/2, img_pt_center2(2)+sy1/2];
                    Screen('DrawTexture',wPtr, input_stim, [], destrect);
                    Screen('DrawTexture',wPtr, input_stim2, [], destrect2);
                end

                % 2. Show Bars and images
                sx=floor(IMAGE_SIZE(1)*disp_scale_causalrating);       sy=floor(IMAGE_SIZE(2)*disp_scale_causalrating);
                % draw bar
                Screen('DrawLine', wPtr, disp_color, bar_pt_start(1), bar_pt_start(2), bar_pt_end(1), bar_pt_end(2),[3]);
                for jj=1:1:length(bar_pt_cursor)
                    Screen('DrawLine', wPtr, disp_color, bar_pt_cursor(jj), bar_pt_start(2)-5, bar_pt_cursor(jj), bar_pt_start(2)+5,[1]);
                    str_block_num=sprintf('%s',num2str(Q{1,q_ind}.cursor_input(jj)));
                    DrawFormattedText(wPtr, str_block_num, bar_pt_cursor(jj)-10, bar_pt_start(2)+25, disp_color);

                end

                img_set_all{block,img_toshow_index(m)}
                block
                img_toshow_index(m)
                input_stim = Screen('MakeTexture', wPtr, img_set_all{block,img_toshow_index(m)});
                img_pt_center=[xpos, bar_pt_start(2)-sy/2-70];
                destrect=[img_pt_center(1)-sx/2,img_pt_center(2)-sy/2,img_pt_center(1)+sx/2,img_pt_center(2)+sy/2];
                Screen('DrawTexture', wPtr, input_stim,[],destrect);
                % 4-2. add 'o' mark at the click pt.
                DrawFormattedText(wPtr, 'o', bar_pt_cursor(current_cursor_selection_ind)-8, bar_pt_start(2)-14, [0 0 0 255]); % add 'o' mark at the click pt.

                % display!
                Screen(wPtr, 'Flip');
                if(ind_ev==1)
                    HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); (5+2*(q_ind-1))]]; % event save
                    clock_time_limit_start=GetSecs;
                end

                % 1. Get cursor (L:37 R:39)
                decision_made=0;
                while(~decision_made)
                    [secs, keyCode] = KbPressWait([], clock_time_limit_start+sec_limit_Q_rating(q_ind)); % if no keyboard in time limit, then go ahead. if pressed earlier, then go ahead.
                    [tmp tmp_key_code]=find(keyCode==1);                zzz=[zzz tmp_key_code];
                    if(tmp_key_code==KEY_L) % L pressed
                        current_cursor_selection_ind=max(1,current_cursor_selection_ind-1);
                        decision_made=1;
                    end
                    if(tmp_key_code==KEY_R) % R pressed
                        current_cursor_selection_ind=min(length(Q{1,q_ind}.cursor_input),current_cursor_selection_ind+1);
                        decision_made=1;
                    end
                    if(tmp_key_code==KEY_Y) % 'y' pressed
                        decision_made=1;            Is_bet=1;
                        bet_score_table(2,ggg_ind)=Q{1,q_ind}.cursor_input(current_cursor_selection_ind);
                        HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); (6+2*(q_ind-1))]]; % event save
                    end
                    if(tmp_key_code==KEY_Q) % 'q' pressed for aborting
                        clear mex
                        decision_made=1;
                    end
                    % check the time limit !@#$
                    if((GetSecs-clock_time_limit_start)>sec_limit_Q_rating(q_ind))
                        decision_made=1;            Is_bet=1;
                        dont_know_cursor_selection=(length(Q{1,q_ind}.cursor_input)+1)/2;
                        bet_score_table(2,ggg_ind)=Q{1,q_ind}.cursor_input(current_cursor_selection_ind);%Q{1,q_ind}.cursor_input(dont_know_cursor_selection);
                        bet_score_table(3,ggg_ind)=-99; % error code
                        HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); -99]]; % event save
                    end
                end


            end

            % a short blank page : to give a sense of the page transition
            Screen(wPtr, 'Flip');
            HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); 20]]; % event save
            WaitSecs(sec_jittered_blank_page);

        end
        HIST_bet_score_table_type2{block,q_ind}=bet_score_table; % same as the above
    end

    if(KEEP_PICTURE_OPT==1)
        %% II. Rating - type1
        % This section handles the type 1 rating phase, where participants rate images by placing points on a triangular grid.

        % Define points of a triangle for the rating display
        tri_pt=zeros(3,2); % edges
        distance_limit=size_triangle+size_triangle*cos(pi/3);
        xpos = round(screenWidth/2);    ypos = round(screenHeight/2)+70;
        tri_pt(1,:)=[xpos,ypos-size_triangle];
        tri_pt(2,:)=[xpos-size_triangle*sin(pi/3),ypos+size_triangle*cos(pi/3)];
        tri_pt(3,:)=[xpos+size_triangle*sin(pi/3),ypos+size_triangle*cos(pi/3)];

        tri_pt_all=zeros(25,2); % all inner points including edges
        tri_pt_all(1,:)=tri_pt(1,:);    tri_pt_all(21,:)=tri_pt(2,:);    tri_pt_all(25,:)=tri_pt(3,:);  tri_pt_all(12,:)=[xpos ypos];
        tri_pt_all(4,:)=[tri_pt_all(1,1) tri_pt_all(1,2)+size_triangle/2];
        tri_pt_all(7,:)=[tri_pt_all(1,1) tri_pt_all(1,2)+3*size_triangle/4];
        tri_pt_all(17,:)=[tri_pt_all(12,1) tri_pt_all(12,2)+size_triangle/4];
        tri_pt_all(2,:)=[tri_pt_all(1,1) tri_pt_all(1,2)+size_triangle/4];
        th_ang=pi/3-atan(cos(pi/6));
        tri_pt_all(3,:)=[tri_pt_all(12,1)-sqrt(7)/4*size_triangle*sin(th_ang) tri_pt_all(12,2)-sqrt(7)/4*size_triangle*cos(th_ang)];
        tri_pt_all(5,:)=[tri_pt_all(12,1)+sqrt(7)/4*size_triangle*sin(th_ang) tri_pt_all(12,2)-sqrt(7)/4*size_triangle*cos(th_ang)];
        tri_pt_all(6,:)=[tri_pt_all(7,1)-size_triangle*tan(pi/3)/4 tri_pt_all(7,2)];
        tri_pt_all(8,:)=[tri_pt_all(7,1)+size_triangle*tan(pi/3)/4 tri_pt_all(7,2)];
        tri_pt_all(9,:)=[tri_pt_all(12,1)-size_triangle*cos(pi/6)/4 tri_pt_all(12,2)-size_triangle*sin(pi/6)/4];
        tri_pt_all(10,:)=[tri_pt_all(12,1)+size_triangle*cos(pi/6)/4 tri_pt_all(12,2)-size_triangle*sin(pi/6)/4];
        th_ang=atan(cos(pi/6))-pi/6;
        tri_pt_all(11,:)=[tri_pt_all(12,1)-sqrt(7)/4*size_triangle*cos(th_ang) tri_pt_all(12,2)+sqrt(7)/4*size_triangle*sin(th_ang)];
        tri_pt_all(13,:)=[tri_pt_all(12,1)+sqrt(7)/4*size_triangle*cos(th_ang) tri_pt_all(12,2)+sqrt(7)/4*size_triangle*sin(th_ang)];
        tri_pt_all(14,:)=[tri_pt_all(12,1)-size_triangle*cos(pi/6)/4 tri_pt_all(12,2)+size_triangle*sin(pi/6)/4];
        tri_pt_all(15,:)=[tri_pt_all(12,1)+size_triangle*cos(pi/6)/4 tri_pt_all(12,2)+size_triangle*sin(pi/6)/4];
        tri_pt_all(16,:)=[tri_pt_all(17,1)-size_triangle*cos(pi/6)/2 tri_pt_all(17,2)];
        tri_pt_all(18,:)=[tri_pt_all(17,1)+size_triangle*cos(pi/6)/2 tri_pt_all(17,2)];
        tri_pt_all(19,:)=[tri_pt_all(12,1)-size_triangle*3*cos(pi/6)/4 tri_pt_all(12,2)+size_triangle*3/8];
        tri_pt_all(20,:)=[tri_pt_all(12,1)+size_triangle*3*cos(pi/6)/4 tri_pt_all(12,2)+size_triangle*3/8];
        tri_pt_all(22,:)=[tri_pt_all(21,1)+size_triangle*cos(pi/6)/2 tri_pt_all(21,2)];
        tri_pt_all(23,:)=[tri_pt_all(12,1) tri_pt_all(12,2)+size_triangle/2];
        tri_pt_all(24,:)=[tri_pt_all(25,1)-size_triangle*cos(pi/6)/2 tri_pt_all(25,2)];
        tri_pt_all=round(tri_pt_all);
        tri_pt_arrange_tbl=[1 2 3 5 4 6 7 8 9 10 12 11 14 15 13 16 17 18 19 20 21:1:25];


        for kk=1:1:size(list_combination,1) % Loop over each combination of images to show

            img_toshow_index=list_combination(kk,randperm(3));

            Is_bet=0;
            bet_score_table=[img_toshow_index; zeros(1,3)];
            bet_money_table=[img_toshow_index; zeros(1,3)];
            Barycentric_coord_table=[img_toshow_index; zeros(1,3)];

            current_cursor_selection_ind_default=find(tri_pt_arrange_tbl==12);
            current_cursor_selection_ind=current_cursor_selection_ind_default;

            ind_ev=0;
            while(~Is_bet)

                ind_ev=ind_ev+1;
                % message
                xpos = round(screenWidth/2);
                ypos = round(screenHeight/2);
                sx = floor(IMAGE_type1(1)*0.7);
                sy = floor(IMAGE_type1(2)*0.7);
                rating = imread([seed_path 'bonus.jpg']);
                input_stim2 = Screen('MakeTexture', wPtr, rating);
                img_pt_center=[xpos, ypos-sy/2-250];
                destrect=[img_pt_center(1)-sx/2, img_pt_center(2)-sy/2, img_pt_center(1)+sx/2, img_pt_center(2)+sy/2];
                Screen('DrawTexture',wPtr, input_stim2, [], destrect);

                % 2. Show triangle
                Screen('DrawLine', wPtr, [0, 0, 0, 255], tri_pt(1,1), tri_pt(1,2), tri_pt(2,1), tri_pt(2,2),[3]);
                Screen('DrawLine', wPtr, [0, 0, 0, 255], tri_pt(2,1), tri_pt(2,2), tri_pt(3,1), tri_pt(3,2),[3]);
                Screen('DrawLine', wPtr, [0, 0, 0, 255], tri_pt(3,1), tri_pt(3,2), tri_pt(1,1), tri_pt(1,2),[3]);
                % show all points of the triangle
                for ttt=1:1:size(tri_pt_all,1)
                    DrawFormattedText(wPtr, '+', tri_pt_all(ttt,1)-8, tri_pt_all(ttt,2)-14, [100 100 100 255]); % add '+' mark at the click pt.
                end
                % show images
                for j=1:1:3
                    selected=img_toshow_index(j);
                    input_stim = Screen('MakeTexture', wPtr, img_set_all{block,selected});
                    sx=floor(IMAGE_SIZE(1)*disp_scale_keepingpicture);       sy=floor(IMAGE_SIZE(2)*disp_scale_keepingpicture);
                    if(j==1)
                        xpos_img = tri_pt(j,1);    ypos_img = tri_pt(j,2)-sy/2;
                    end
                    if(j==2)
                        xpos_img = tri_pt(j,1)-sx/2;    ypos_img = tri_pt(j,2)+sy/2;
                    end
                    if(j==3)
                        xpos_img = tri_pt(j,1)+sx/2;    ypos_img = tri_pt(j,2)+sy/2;
                    end
                    destrect=[xpos_img-sx/2,ypos_img-sy/2,xpos_img+sx/2,ypos_img+sy/2];
                    Screen('DrawTexture', wPtr, input_stim,[],destrect);
                end

                x=tri_pt_all(tri_pt_arrange_tbl(current_cursor_selection_ind),1);   y=tri_pt_all(tri_pt_arrange_tbl(current_cursor_selection_ind),2);
                DrawFormattedText(wPtr, 'o', x-8,y-14, [0 0 255 255]); % add 'o' mark at the click pt.

                % 3-1. Barycentric_coordinate_system
                denorm=(tri_pt(2,2)-tri_pt(3,2))*(tri_pt(1,1)-tri_pt(3,1))+(tri_pt(3,1)-tri_pt(2,1))*(tri_pt(1,2)-tri_pt(3,2));
                Barycentric_coord_table(2,1)=((tri_pt(2,2)-tri_pt(3,2))*(x-tri_pt(3,1))+(tri_pt(3,1)-tri_pt(2,1))*(y-tri_pt(3,2)))/denorm;
                Barycentric_coord_table(2,1)=max(Barycentric_coord_table(2,1),0);   Barycentric_coord_table(2,1)=min(Barycentric_coord_table(2,1),1);
                Barycentric_coord_table(2,2)=((tri_pt(3,2)-tri_pt(1,2))*(x-tri_pt(3,1))+(tri_pt(1,1)-tri_pt(3,1))*(y-tri_pt(3,2)))/denorm;
                Barycentric_coord_table(2,2)=max(Barycentric_coord_table(2,2),0);   Barycentric_coord_table(2,2)=min(Barycentric_coord_table(2,2),1);
                Barycentric_coord_table(2,3)=1-Barycentric_coord_table(2,1)-Barycentric_coord_table(2,2);
                Barycentric_coord_table(2,:)=abs(Barycentric_coord_table(2,:));

                % 4-3. compute and add betting values from distance table
                bet_score_table(2,:)=Barycentric_coord_table(2,:);
                for j=1:1:3 % add the betting values to display
                    sx=floor(IMAGE_SIZE(1)*disp_scale_keepingpicture);       sy=floor(IMAGE_SIZE(2)*disp_scale_keepingpicture);
                    if(j==1)
                        xpos_score_disp = tri_pt(j,1)-40;    ypos_score_disp = tri_pt(j,2)-sy-35;
                    end
                    if(j==2)
                        xpos_score_disp = tri_pt(j,1)-sx;    ypos_score_disp = tri_pt(j,2)+sy+10;
                    end
                    if(j==3)
                        xpos_score_disp = tri_pt(j,1);    ypos_score_disp = tri_pt(j,2)+sy+10;
                    end
                    destrect=[xpos_score_disp-sx/2,ypos_score_disp-sy/2,xpos_score_disp+sx/2,ypos_score_disp+sy/2];
                    bet_money_table(2,j)=double(money_given*bet_score_table(2,j));
                    str_score=sprintf('%01.2f',bet_money_table(2,j));
                    DrawFormattedText(wPtr, str_score, xpos_score_disp, ypos_score_disp);
                end

                % display!
                Screen(wPtr, 'Flip');
                if(ind_ev==1)
                    HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); 9]]; % event save
                    clock_time_limit_start=GetSecs;
                end

                % 1. Get cursor (L:37 R:39)
                decision_made=0;
                while(~decision_made)
                    [secs, keyCode] = KbPressWait([], clock_time_limit_start+sec_limit_Bayesian_rating); % if no keyboard in time limit, then go ahead. if pressed earlier, then go ahead.
                    [tmp tmp_key_code]=find(keyCode==1);                zzz=[zzz tmp_key_code];
                    if(tmp_key_code==KEY_L) % L pressed
                        current_cursor_selection_ind=max(1,current_cursor_selection_ind-1);
                        decision_made=1;
                    end
                    if(tmp_key_code==KEY_R) % R pressed
                        current_cursor_selection_ind=min(size(tri_pt_all,1),current_cursor_selection_ind+1);
                        decision_made=1;
                    end
                    if(tmp_key_code==KEY_Y) % 'y' pressed
                        decision_made=1;            Is_bet=1;
                        bet_score_table(2,:)=Barycentric_coord_table(2,:);
                        HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); 10]]; % event save
                    end
                    if(tmp_key_code==KEY_N) % 'n' pressed - reset
                        decision_made=1;            Is_bet=0;
                        current_cursor_selection_ind=current_cursor_selection_ind_default;
                    end
                    if(tmp_key_code==KEY_Q) % 'q' pressed for aborting
                        clear mex
                        decision_made=1;
                    end
                    % check the time limit
                    if((GetSecs-clock_time_limit_start)>sec_limit_Bayesian_rating)
                        decision_made=1;            Is_bet=1;
                        dont_know_cursor_selection=(length(Q{1,q_ind}.cursor_input)+1)/2;
                        bet_score_table(2,:)=Barycentric_coord_table(2,:);%[1/3 1/3 1/3];
                        bet_score_table(3,:)=-99; % error code
                        HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); -99]]; % event save
                    end
                end
            end

            % a short blank page : to give a sense of the page transition
            Screen(wPtr, 'Flip');
            HIST_event_info0=[HIST_event_info0 [block; 0; 0; (GetSecs-session_clock_start); (GetSecs-block_clock_start); 20]]; % event save
            WaitSecs(sec_jittered_blank_page);

            % Save the current trial’s betting data
            HIST_Barycentric_coord_table_type1{block,kk}=Barycentric_coord_table; % sum to one
            HIST_bet_score_table_type1{block,kk}=bet_score_table; % same as the above
            HIST_bet_money_table_type1{block,kk}=bet_money_table; % moeny bet amount

        end

    end

    block_kind = 1;
    [r_tmp c_tmp]=find(Tot_img_info(3,:)==index_num);
    [r_tmp2 c_tmp2]=find(Tot_img_info(4,c_tmp)==block);
    absolute_index=c_tmp(c_tmp2);
    for hhh=1:1:length(absolute_index)
        novelty_lev=Tot_img_info(5,absolute_index(hhh)); % read out novelty level
        % update hedonic ratings
        [r_tmp3 c_tmp3]=find(HIST_bet_score_table_type2{block,1}(1,:)==novelty_lev);
        Tot_img_info(6,absolute_index(hhh))=HIST_bet_score_table_type2{block,1}(2,c_tmp3);
        % update causal ratings
        [r_tmp3 c_tmp3]=find(HIST_bet_score_table_type2{block,2}(1,:)==novelty_lev);
        Tot_img_info(7,absolute_index(hhh))=HIST_bet_score_table_type2{block,2}(2,c_tmp3);
        % update bonus round ratings
        [r_tmp3 c_tmp3]=find(HIST_bet_score_table_type1{block,1}(1,:)==novelty_lev);
        Tot_img_info(8,absolute_index(hhh))=HIST_bet_score_table_type1{block,1}(2,c_tmp3);
        % update nov-pos=0, nov-neg=1
        Tot_img_info(9,absolute_index(hhh))=block_kind;
    end
    %% Update HIST_event_info (overwrite at each block)
    HIST_event_info{1,index_num}=HIST_event_info0;


    %% save the (updated) image usage matrix (overwriting)
    file_imgind_sv_name=[EXP_NAME '_image_usage_info.mat'];
    file_name_sv=[save_path file_imgind_sv_name];
    % eval(['save ' file_name_sv ' Tot_img_info Tot_img_info_Tag']);
    save(file_name_sv,'Tot_img_info','Tot_img_info_Tag','HIST_event_info','HIST_event_info_Tag');

end


%% Ending message
str_end=sprintf('- Our experiments is over. Press any key to quit. -');
DrawFormattedText(wPtr, str_end, 'center', 'center');
Screen(wPtr, 'Flip');
KbWait; % temporarily disabled for test APR 21

%% save all data

data_save.HIST_bet_score_table_type2=cell(Tot_block,1);
for k=1:1:Tot_block
    if(KEEP_PICTURE_OPT==1)
        [tmp dd]=sort(HIST_bet_money_table_type1{k,1}(1,:)); % sort by index
        data_save.HIST_bet_money_table_type1{k,1}(:,:)=HIST_bet_money_table_type1{k,1}(:,dd);
        [tmp dd]=sort(HIST_bet_score_table_type1{k,1}(1,:)); % sort by index
        data_save.HIST_bet_score_table_type1{k,1}=HIST_bet_score_table_type1{k,1}(:,dd);
        [tmp dd]=sort(HIST_Barycentric_coord_table_type1{k,1}(1,:)); % sort by index
        data_save.HIST_Barycentric_coord_table_type1{k,1}=HIST_Barycentric_coord_table_type1{k,1}(:,dd);
    end
    [tmp dd]=sort(HIST_bet_score_table_type2{k,1}(1,:)); % sort by index
    data_save.HIST_bet_score_table_type2{k,1}=HIST_bet_score_table_type2{k,1}(:,dd);
end
data.HIST_mapping_index2filenum=HIST_current_pool;
data_save.HIST_schedule=HIST_schedule;
data_save.HIST_reward=HIST_reward;
data_save.EXP_NAME=EXP_NAME;
data_save.Tot_block=Tot_block;
data_save.Tot_trial=Tot_trial;
data_save.Tot_trial_s=Tot_trial_s;
data_save.tot_num_img_per_block=tot_num_img_per_block;

data_save.img_set=img_set;
data_save.img0_set=img0_set;

data_save.sec_stim_interval=sec_stim_interval;%1.5; %(sec)
data_save.earliest_novel_stimulus_show_in_block=earliest_novel_stimulus_show_in_block; % 0: any trial in block, 1: last?

data_save.reward_mag=reward_mag;
data_save.reward_prob=reward_prob;

data_save.money_given=money_given; % point
data_save.img_present_ratio=img_present_ratio; % novelty level :: size = tot_num_img_per_block-1.

%% save the (updated) image usage matrix (overwriting)
file_imgind_sv_name=[EXP_NAME '_image_usage_info.mat'];
file_name_sv=[save_path file_imgind_sv_name];
% eval(['save ' file_name_sv ' Tot_img_info Tot_img_info_Tag']);
save(file_name_sv,'Tot_img_info','Tot_img_info_Tag','HIST_event_info','HIST_event_info_Tag');

%% save all variables
if(strcmp(session_opt,'pre')==1)
    file_sv_name=[EXP_NAME sprintf('_pre_%d.mat',index_num)];
elseif (strcmp(session_opt,'main')==1)
    file_sv_name=[EXP_NAME sprintf('_main_%d.mat',index_num)];
end

file_name_sv=[save_path file_sv_name];
disp(file_sv_name);
save(file_name_sv,'*');

%% finish all
Screen('CloseAll');
clear mex
% clear Screen

disp('########################################################')
str_end1=sprintf('### session%d is done ############################',index_num);
disp(str_end1);
str_end2=sprintf('### next session = %d ############################',index_num+1);
disp(str_end2);
disp('########################################################')


% display the number of response failure
missed_count = length(find(HIST_event_info{1,index_num}(6,:)==-99));
disp(sprintf('- # of response failure in this session = %d. (will be penalized) ',missed_count));

% Save data input for GAN novelty detection
save_input_gan_novelty(EXP_NAME, index_num);

% Perform cleanup and execute external training and generation commands
[status, message, messageid] = rmdir('asset', 's');
[val21, val22] = system(['main_train_cnn.exe ' EXP_NAME ' ' int2str(index_num)]);
[val21, val22] = system(['main_generate_cnn.exe ' EXP_NAME ' ' int2str(index_num)]);
[val21, val22] = system(['main_train_gan.exe ' EXP_NAME ' ' int2str(index_num)]);
[val21, val22] = system(['main_generate_gan.exe ' EXP_NAME ' ' int2str(index_num)]);
output_info = 1; % Set output info to indicate success
[status, message, messageid] = rmdir('asset', 's');

end


