%% preprocess_AF_vs_NSR.m
% Preprocess ECG data for AF vs NSR classification
% Matching TransMixer-AF style (Dataset-1 PhysioNet 2017)
% No Wavelet Toolbox needed

clear; clc; close all;

%% Load reference labels (REERENCE.csv must be in this folder)
T = readtable('REFERENCE.csv', 'ReadVariableNames', false);
refIDs   = string(T.Var1);
refLabel = string(T.Var2);

% Indices of Normal & AF
idxNSR = find(refLabel == "N");
idxAF  = find(refLabel == "A");

fprintf("Total NSR: %d, AF: %d\n", numel(idxNSR), numel(idxAF));

%% Balanced subset (same count NSR & AF)
numAF  = numel(idxAF);
rng(0); % reproducible
idxNSR_sel = idxNSR(randperm(numel(idxNSR), numAF));  % select equal NSR

subsetIdx = [idxAF; idxNSR_sel];
subsetIDs = refIDs(subsetIdx);
subsetLab = refLabel(subsetIdx);

numTrials = numel(subsetIdx);
fprintf("Processing %d signals: %d AF, %d NSR\n", ...
    numTrials, numAF, numAF);

%% Preprocessing settings
fs = 300;                 % Sampling freq (Hz)
targetLength = 9000;      % 30s → 30*300=9000 samples

% Bandpass filter
[b,a] = butter(4, [0.5 40] / (fs/2), 'bandpass');

% Median window for baseline (about 0.6s)
medWin = round(0.6 * fs);
if mod(medWin,2) == 0, medWin = medWin + 1; end

% Preallocate
X = zeros(numTrials, targetLength);
Y = zeros(numTrials, 1);
recIDs = strings(numTrials,1);

progressStep = max(1, floor(numTrials/20));

%% Loop through selected signals
for k = 1:numTrials
    fname = subsetIDs(k) + ".mat";
    recIDs(k) = subsetIDs(k);

    if ~isfile(fname)
        warning("Missing file: %s", fname);
        continue;
    end

    S = load(fname);
    sig = S.val(:)'; % make row

    % Truncate or zero-pad to 9000 samples
    if length(sig) > targetLength
        sig = sig(1:targetLength);
    elseif length(sig) < targetLength
        sig = [sig zeros(1, targetLength - length(sig))];
    end

    % Bandpass filter
    sig = filtfilt(b,a,sig);

    % Baseline removal using median filter
    baseline = medfilt1(sig, medWin);
    sig = sig - baseline;

    % Normalize (z-score)
    mu = mean(sig); sigma = std(sig);
    if sigma > 0
        sig = (sig - mu)/sigma;
    else
        sig = sig - mu;
    end

    X(k,:) = sig;
    % Assign labels: 1 = AF, 0 = NSR
    Y(k) = double(subsetLab(k) == "A");

    if mod(k, progressStep) == 0 || k == numTrials
        fprintf("Processed %d/%d (%.1f%%)\n", k, numTrials, 100*k/numTrials);
    end
end

%% Save results
save('preprocessed_ECG_all.mat', 'X','Y','recIDs','fs', ...
    'subsetIdx','subsetIDs','subsetLab','-v7.3');

fprintf("\n✔ Finished! Saved %d trials x %d samples.\n", size(X,1), size(X,2));


