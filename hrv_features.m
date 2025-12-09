%% extract_hrv_features.m
% Use RR intervals (from rr_data.mat) to compute derived HRV features
% Features per recording:
% [ meanRR, meanHR, SDNN, RMSSD, pNN50, IQR ]

clear; clc;

% This file was created earlier by compute_rr_intervals.m
load rr_data.mat    % gives rr_intervals, Y, recIDs, fs

numTrials = numel(rr_intervals);

% Preallocate: rows = signals, cols = features
% 1: meanRR (s)
% 2: meanHR (bpm)
% 3: SDNN (s)
% 4: RMSSD (s)
% 5: pNN50 (0–1)
% 6: IQR (s)
HRV = nan(numTrials, 6);

for i = 1:numTrials
    rr = rr_intervals{i};      % RR intervals for this recording (in seconds)

    % skip if not enough intervals
    if numel(rr) < 2
        continue
    end

    % --- basic stats ---
    meanRR = mean(rr);
    meanHR = 60 ./ meanRR;             % bpm

    sdnn   = std(rr);                  % overall variability

    dRR    = diff(rr);
    rmssd  = sqrt(mean(dRR.^2));       % short-term variability

    pnn50  = sum(abs(dRR) > 0.05) / numel(dRR);  % >50 ms changes

    iqrRR  = iqr(rr);                  % interquartile range

    HRV(i,:) = [meanRR, meanHR, sdnn, rmssd, pnn50, iqrRR];
end

% Remove rows where HRV couldn't be computed
valid = all(~isnan(HRV), 2);
X_hrv = HRV(valid, :);
Y_hrv = Y(valid);
recIDs_hrv = recIDs(valid);

save('derived_HRV_features.mat', 'X_hrv','Y_hrv','recIDs_hrv');

fprintf('✔ HRV feature extraction done.\n');
fprintf('Valid recordings: %d / %d\n', size(X_hrv,1), numTrials);

%% now we can see the number of recordings done
numAF = sum(Y_hrv == 1);
numNSR = sum(Y_hrv == 0);

fprintf("NSR: %d | AF: %d\n", numNSR, numAF);

%% HRV features plot
load derived_HRV_features.mat  % X_hrv, Y_hrv

featureNames = {'meanRR','HR','SDNN','RMSSD','pNN50','IQR'};
numFeatures = size(X_hrv,2);

figure;
for f = 1:numFeatures
    subplot(2,3,f);
    hold on;
    plot(X_hrv(Y_hrv==0,f), 'b.', 'MarkerSize', 6);
    plot(X_hrv(Y_hrv==1,f), 'r.', 'MarkerSize', 6);
    title(featureNames{f});
    xlabel('Recording Index');
    ylabel('Value');
    legend({'NSR','AF'});
    grid on;
end
sgtitle('Derived HRV Features Comparison - NSR vs AF');

