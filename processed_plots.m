load preprocessed_ECG_all.mat   % loads X, Y, recIDs, fs

numExamples = 4;  % number of AF + NSR to plot

% Plot AF examples
figure;
af_idx = find(Y == 1);
for i = 1:numExamples
    subplot(numExamples,1,i);
    plot((1:size(X,2))/fs, X(af_idx(i),:));
    title(sprintf("AF Example %d - %s", i, recIDs(af_idx(i))));
    xlabel("Time (s)");
    ylabel("Amplitude (normalized)");
end
sgtitle("AF Signals (Normalized ECG after Preprocessing)");

% Plot NSR examples
figure;
nsr_idx = find(Y == 0);
for i = 1:numExamples
    subplot(numExamples,1,i);
    plot((1:size(X,2))/fs, X(nsr_idx(i),:));
    title(sprintf("NSR Example %d - %s", i, recIDs(nsr_idx(i))));
    xlabel("Time (s)");
    ylabel("Amplitude (normalized)");
end
sgtitle("NSR Signals (Normalized ECG after Preprocessing)");
