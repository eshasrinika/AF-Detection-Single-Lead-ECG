%% Beat-based CNN+BiLSTM AF vs NSR Detection – Simple Baseline
clear; clc; close all;
%% 1) Load beat-level dataset
% Your .mat has:
%   Xbeats    : cell array of beats  (each: [numFeatures x timeSamples])
%   YbeatsCat : categorical labels   ("NSR","AF")
load('beat_dataset.mat','Xbeats','YbeatsCat');
X = Xbeats;
Y = YbeatsCat;           % already categorical
numSamples = numel(X);
%% 2) Train/Val/Test split
rng(3);                  % reproducible
idx = randperm(numSamples);
nTrain = round(0.7 * numSamples);
nVal   = round(0.15 * numSamples);
nTest  = numSamples - nTrain - nVal;
idxTrain = idx(1:nTrain);
idxVal   = idx(nTrain+1 : nTrain+nVal);
idxTest  = idx(nTrain+nVal+1 : end);
XTrain = X(idxTrain);
YTrain = Y(idxTrain);
XVal   = X(idxVal);
YVal   = Y(idxVal);
XTest  = X(idxTest);
YTest  = Y(idxTest);
%% 3) Convert to single precision (helps with memory)
for i = 1:numel(XTrain)
    XTrain{i} = single(XTrain{i});
end
for i = 1:numel(XVal)
    XVal{i} = single(XVal{i});
end
for i = 1:numel(XTest)
    XTest{i} = single(XTest{i});
end
%% 4) Define simple CNN + BiLSTM network
% Each beat: [numFeatures x timeSamples]
inputSize  = size(XTrain{1},1);
classNames = categories(YTrain);
numClasses = numel(classNames);
filterSize1  = 3;
numFilters1  = 16;
filterSize2  = 3;
numFilters2  = 32;
layers = [
    sequenceInputLayer(inputSize,"Name","input")
    % --- 1D CNN feature extractor (no pooling) ---
    convolution1dLayer(filterSize1,numFilters1,"Padding","same","Name","conv1")
    reluLayer("Name","relu1")
    convolution1dLayer(filterSize2,numFilters2,"Padding","same","Name","conv2")
    reluLayer("Name","relu2")
    % --- BiLSTM sequence modelling ---
    bilstmLayer(64,"OutputMode","last","Name","bilstm")
    % --- Classifier ---
    fullyConnectedLayer(numClasses,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","output","Classes",classNames)
];
%% 5) Training options (simple + memory-friendly)
options = trainingOptions("adam", ...
    "ExecutionEnvironment","cpu", ...   % avoid GPU OOM
    "MiniBatchSize",32, ...            % if OOM, reduce to 16 or 8
    "MaxEpochs",25, ...
    "Shuffle","every-epoch", ...
    "ValidationData",{XVal, YVal}, ...
    "ValidationFrequency",50, ...
    "GradientThreshold",1, ...
    "Verbose",true, ...
    "Plots","training-progress");
%% 6) Train network
net = trainNetwork(XTrain, YTrain, layers, options);
%% 7) Evaluate on test set
[YPredTest, scoresTest] = classify(net, XTest, "MiniBatchSize",64);
% Confusion matrix – standard decision
figure;
confusionchart(YTest, YPredTest, ...
    "RowSummary","row-normalized", ...
    "ColumnSummary","column-normalized");
title("Beat-based CNN+BiLSTM AF Detection - Confusion Matrix");
% Accuracy & AF recall
classNamesNet = net.Layers(end).Classes;   % categorical
cm = confusionmat(YTest, YPredTest, "Order", classNamesNet);
accuracy = sum(diag(cm)) / sum(cm(:));
idxAF = find(classNamesNet == 'AF');
tpAF  = cm(idxAF, idxAF);
fnAF  = sum(cm(idxAF, :)) - tpAF;
recallAF = tpAF / (tpAF + fnAF);
fprintf("\n=== CNN+BiLSTM – Standard decision ===\n");
fprintf("Overall accuracy: %.2f %%\n", accuracy*100);
fprintf("AF recall:        %.2f %%\n", recallAF*100);
