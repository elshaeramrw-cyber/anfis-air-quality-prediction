clear; clc; close all;
rng(42);

%% ===================== SETTINGS =====================
filename = 'AirQualityUCI.csv';

numRuns = 10;
trainRatio = 0.70;
valRatio   = 0.15;

smoothTarget = true;
smoothWindow = 5;

savePlots = true;

% Faster than GridPartition. Rule count = number of clusters.
clusterList = [6 8 10];     % Try [4 6 8 10 12] if you want more tuning
epochsList  = [100 150];    % Smaller than 300 for speed
stepSize    = 0.01;

%% ===================== LOAD DATA =====================
data_table = readtable(filename, ...
    'FileType', 'text', ...
    'Delimiter', ';', ...
    'ReadVariableNames', false, ...
    'TreatAsMissing', '-200');

fprintf('Original rows: %d\n', height(data_table));

% AirQualityUCI has an extra empty column sometimes
data_table = data_table(:, 1:15);

varNames = {'Date','Time','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT', ...
            'PT08_S2_NMHC','NOx_GT','PT08_S3_NOx','NO2_GT', ...
            'PT08_S4_NO2','PT08_S5_O3','T','RH','AH'};

data_table.Properties.VariableNames = varNames;

numericVars = varNames(3:end);

for i = 1:length(numericVars)
    colName = numericVars{i};
    colData = data_table.(colName);

    if iscell(colData) || isstring(colData)
        colData = strrep(string(colData), ',', '.');
        colData = str2double(colData);
    else
        colData = double(colData);
    end

    data_table.(colName) = fillmissing(colData, 'linear', 'EndValues', 'nearest');
end

remaining_missing = sum(any(ismissing(data_table{:, numericVars}), 2));
fprintf('Rows with missing after interpolation: %d\n', remaining_missing);

%% ===================== INPUTS AND TARGET =====================
inputNames = {'PT08_S1_CO','PT08_S2_NMHC','PT08_S3_NOx', ...
              'PT08_S4_NO2','PT08_S5_O3','T','RH','AH'};

inputsRaw = data_table{:, inputNames};
targetRaw = data_table{:, 'C6H6_GT'};

if smoothTarget
    targetRaw = movmean(targetRaw, smoothWindow);
end

valid = all(~isnan(inputsRaw), 2) & ~isnan(targetRaw);

if sum(~valid) > 0
    fprintf('Removing %d invalid rows.\n', sum(~valid));
    inputsRaw = inputsRaw(valid, :);
    targetRaw = targetRaw(valid);
end

fprintf('Final dataset size: %d rows\n', size(inputsRaw, 1));

%% ===================== STORAGE =====================
trainRMSE = zeros(numRuns,1);
valRMSE   = zeros(numRuns,1);
testRMSE  = zeros(numRuns,1);

testMAE = zeros(numRuns,1);
testR2  = zeros(numRuns,1);
accuracy = zeros(numRuns,1);

bestOverallValRMSE = inf;
bestOverallFis = [];
bestScale = struct();

fprintf('\nStarting ANFIS training...\n');

%% ===================== RUNS =====================
for run = 1:numRuns
    fprintf('\n========== RUN %d/%d ==========\n', run, numRuns);

    nSamples = size(inputsRaw, 1);
    idx = randperm(nSamples);

    nTrain = round(trainRatio * nSamples);
    nVal   = round(valRatio * nSamples);

    idxTrain = idx(1:nTrain);
    idxVal   = idx(nTrain+1 : nTrain+nVal);
    idxTest  = idx(nTrain+nVal+1 : end);

    XTrainRaw = inputsRaw(idxTrain, :);
    YTrainRaw = targetRaw(idxTrain);

    XValRaw = inputsRaw(idxVal, :);
    YValRaw = targetRaw(idxVal);

    XTestRaw = inputsRaw(idxTest, :);
    YTestRaw = targetRaw(idxTest);

    % Normalize using training data only
    [XTrain, inputMin, inputRange] = normalizeTrain(XTrainRaw);
    XVal  = normalizeApply(XValRaw, inputMin, inputRange);
    XTest = normalizeApply(XTestRaw, inputMin, inputRange);

    [YTrain, targetMin, targetRange] = normalizeTrain(YTrainRaw);
    YVal  = normalizeApply(YValRaw, targetMin, targetRange);
    YTest = normalizeApply(YTestRaw, targetMin, targetRange);

    trainData = [XTrain YTrain];
    valData   = [XVal YVal];

    bestRunValRMSE = inf;
    bestRunFis = [];
    bestRunName = '';

    %% Tune small set of FCM cluster counts and epochs
    for c = 1:length(clusterList)
        for e = 1:length(epochsList)
            numClusters = clusterList(c);
            epochs = epochsList(e);

            fisOptions = genfisOptions('FCMClustering');
            fisOptions.NumClusters = numClusters;
            fisOptions.Verbose = false;

            initialFis = genfis(XTrain, YTrain, fisOptions);

            anfisOpt = anfisOptions( ...
                'InitialFIS', initialFis, ...
                'EpochNumber', epochs, ...
                'InitialStepSize', stepSize, ...
                'ValidationData', valData, ...
                'DisplayANFISInformation', false, ...
                'DisplayErrorValues', false, ...
                'DisplayStepSize', false, ...
                'DisplayFinalResults', false);

            [trainedFis, trainError, ~, chkFis, chkError] = anfis(trainData, anfisOpt);

            if ~isempty(chkFis)
                candidateFis = chkFis;
                candidateValRMSE = min(chkError);
            else
                candidateFis = trainedFis;
                YValPred = evalfis(candidateFis, XVal);
                candidateValRMSE = sqrt(mean((YValPred - YVal).^2));
            end

            fprintf('Clusters=%2d, Epochs=%3d, ValRMSE=%.4f\n', ...
                numClusters, epochs, candidateValRMSE);

            if candidateValRMSE < bestRunValRMSE
                bestRunValRMSE = candidateValRMSE;
                bestRunFis = candidateFis;
                bestRunName = sprintf('FCM clusters=%d, epochs=%d', numClusters, epochs);
            end
        end
    end

    fprintf('Best model for run %d: %s\n', run, bestRunName);

    %% Predictions
    YTrainPred = evalfis(bestRunFis, XTrain);
    YValPred   = evalfis(bestRunFis, XVal);
    YTestPred  = evalfis(bestRunFis, XTest);

    trainRMSE(run) = sqrt(mean((YTrainPred - YTrain).^2));
    valRMSE(run)   = sqrt(mean((YValPred - YVal).^2));
    testRMSE(run)  = sqrt(mean((YTestPred - YTest).^2));

    % Convert predictions back to original benzene scale
    YTestPredOrig = denormalizeApply(YTestPred, targetMin, targetRange);
    YTestOrig = YTestRaw;

    testMAE(run) = mean(abs(YTestPredOrig - YTestOrig));

    ssRes = sum((YTestOrig - YTestPredOrig).^2);
    ssTot = sum((YTestOrig - mean(YTestOrig)).^2);
    testR2(run) = 1 - (ssRes / ssTot);

    %% Classification using train-set thresholds only
    lowThresh  = prctile(YTrainRaw, 33);
    highThresh = prctile(YTrainRaw, 66);

    actualClass = classifyAirQuality(YTestOrig, lowThresh, highThresh);
    predClass   = classifyAirQuality(YTestPredOrig, lowThresh, highThresh);

    accuracy(run) = mean(predClass == actualClass) * 100;

    fprintf('Run %2d: TrainRMSE=%.4f, ValRMSE=%.4f, TestRMSE=%.4f, Acc=%.2f%%, R2=%.4f\n', ...
        run, trainRMSE(run), valRMSE(run), testRMSE(run), accuracy(run), testR2(run));

    disp('Confusion matrix:');
    disp(confusionmat(actualClass, predClass));

    %% Keep best validation model overall
    if bestRunValRMSE < bestOverallValRMSE
        bestOverallValRMSE = bestRunValRMSE;
        bestOverallFis = bestRunFis;

        bestScale.inputMin = inputMin;
        bestScale.inputRange = inputRange;
        bestScale.targetMin = targetMin;
        bestScale.targetRange = targetRange;
    end

    %% Optional plot for this run only if needed
    if savePlots
        figure('Visible', 'off');
        scatter(YTestOrig, YTestPredOrig, 10, 'filled');
        hold on;
        plot([min(YTestOrig), max(YTestOrig)], ...
             [min(YTestOrig), max(YTestOrig)], 'r--', 'LineWidth', 1.5);
        xlabel('Actual Benzene');
        ylabel('Predicted Benzene');
        title(sprintf('Run %d: Actual vs Predicted', run));
        grid on;
        saveas(gcf, sprintf('scatter_run_%02d.png', run));
        close(gcf);
    end
end

%% ===================== FINAL RESULTS =====================
fprintf('\n========== RESULTS OVER %d RUNS ==========\n', numRuns);
fprintf('Training RMSE   : %.4f ± %.4f\n', mean(trainRMSE), std(trainRMSE));
fprintf('Validation RMSE : %.4f ± %.4f\n', mean(valRMSE), std(valRMSE));
fprintf('Test RMSE       : %.4f ± %.4f\n', mean(testRMSE), std(testRMSE));
fprintf('Test MAE        : %.4f ± %.4f\n', mean(testMAE), std(testMAE));
fprintf('Test R2         : %.4f ± %.4f\n', mean(testR2), std(testR2));
fprintf('Accuracy        : %.2f%% ± %.2f%%\n', mean(accuracy), std(accuracy));

%% ===================== FINAL VISUALIZATION =====================
inputsNormAll = normalizeApply(inputsRaw, bestScale.inputMin, bestScale.inputRange);
allPredNorm = evalfis(bestOverallFis, inputsNormAll);
allPredOrig = denormalizeApply(allPredNorm, bestScale.targetMin, bestScale.targetRange);

figure('Name', 'Best ANFIS Structure');
plotfis(bestOverallFis);

figure('Name', 'Membership Functions of First Input');
plotmf(bestOverallFis, 'input', 1);

figure('Name', 'Actual vs Predicted Benzene');
scatter(targetRaw, allPredOrig, 10, 'filled');
hold on;
plot([min(targetRaw), max(targetRaw)], ...
     [min(targetRaw), max(targetRaw)], 'r--', 'LineWidth', 2);
xlabel('Actual Benzene');
ylabel('Predicted Benzene');
title(sprintf('Best ANFIS Model: Avg Test RMSE = %.4f', mean(testRMSE)));
grid on;

%% ===================== HELPER FUNCTIONS =====================
function [Xnorm, Xmin, Xrange] = normalizeTrain(X)
    Xmin = min(X, [], 1);
    Xmax = max(X, [], 1);
    Xrange = Xmax - Xmin;
    Xrange(Xrange == 0) = 1;
    Xnorm = (X - Xmin) ./ Xrange;
end

function Xnorm = normalizeApply(X, Xmin, Xrange)
    Xnorm = (X - Xmin) ./ Xrange;
end

function Xorig = denormalizeApply(Xnorm, Xmin, Xrange)
    Xorig = Xnorm .* Xrange + Xmin;
end

function classLabels = classifyAirQuality(y, lowThresh, highThresh)
    classLabels = zeros(size(y));
    classLabels(y <= lowThresh) = 1;
    classLabels(y > lowThresh & y < highThresh) = 2;
    classLabels(y >= highThresh) = 3;
end
