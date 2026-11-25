clear; clc; close all;
rng(42);
tic;

%% Load data
filename = 'AirQualityUCI.csv';
data_table = readtable(filename, 'FileType', 'text', 'Delimiter', ';', ...
    'ReadVariableNames', false, 'TreatAsMissing', '-200');

data_table = data_table(:, 1:15);
varNames = {'Date','Time','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT', ...
            'PT08_S2_NMHC','NOx_GT','PT08_S3_NOx','NO2_GT','PT08_S4_NO2', ...
            'PT08_S5_O3','T','RH','AH'};
data_table.Properties.VariableNames = varNames;

%% Parse numeric columns
numericVars = setdiff(varNames, {'Date','Time'});
for i = 1:length(numericVars)
    colName = numericVars{i};
    colData = data_table.(colName);
    if iscell(colData)
        colData = strrep(colData, ',', '.');
        data_table.(colName) = str2double(colData);
    else
        data_table.(colName) = double(colData);
    end
    data_table.(colName) = fillmissing(data_table.(colName), 'linear');
end

%% Prepare data — no movmean (avoids temporal leakage)
all_inputs = data_table{:, {'PT08_S1_CO','PT08_S2_NMHC','PT08_S3_NOx','PT08_S4_NO2','PT08_S5_O3'}};
target = data_table{:, 'C6H6_GT'};

valid = all(~isnan(all_inputs),2) & ~isnan(target);
if sum(~valid) > 0
    fprintf('Removing %d rows with NaNs.\n', sum(~valid));
    all_inputs = all_inputs(valid,:);
    target = target(valid);
end
fprintf('Dataset: %d rows\n', size(all_inputs,1));

%% Feature selection — keep only the most correlated inputs
inputNames = {'PT08_S1_CO','PT08_S2_NMHC','PT08_S3_NOx','PT08_S4_NO2','PT08_S5_O3'};
corrs = zeros(1,5);
for i = 1:5
    r = corrcoef(all_inputs(:,i), target);
    corrs(i) = abs(r(1,2));
    fprintf('  |%s| = %.4f\n', inputNames{i}, corrs(i));
end
[~, bestIdx] = sort(corrs, 'descend');

nFeatures = 3;                          % 3 inputs → much fewer rules than 5
selIdx = bestIdx(1:nFeatures);
inputs = all_inputs(:, selIdx);
fprintf('Selected: %s\n', strjoin(inputNames(selIdx), ', '));

%% Normalize
in_min = min(inputs);  in_max = max(inputs);
inputs_norm = (inputs - in_min) ./ (in_max - in_min);
t_min = min(target);   t_max = max(target);
target_norm = (target - t_min) ./ (t_max - t_min);

%% Parameters
numRuns = 10;
trainRatio = 0.7;
valRatio   = 0.15;
epochs     = 200;           % slightly fewer epochs = faster
stepSize   = 0.05;          % tuned
clusterRad = 0.45;          % subtractive-clustering radius

trainRMSE = zeros(numRuns,1);
valRMSE   = zeros(numRuns,1);
testRMSE  = zeros(numRuns,1);
accuracy  = zeros(numRuns,1);

bestValRMSE = inf;
bestFis = [];
bestRun = 0;

nSamples = size(inputs_norm,1);
fprintf('\n%d runs, %d features, radius=%.2f\n', numRuns, nFeatures, clusterRad);

%% Main loop
for run = 1:numRuns
    idx = randperm(nSamples);
    nTrain = round(trainRatio * nSamples);
    nVal   = round(valRatio * nSamples);

    iTr = idx(1:nTrain);
    iVa = idx(nTrain+1 : nTrain+nVal);
    iTe = idx(nTrain+nVal+1 : end);

    Xtr = inputs_norm(iTr,:);  Ytr = target_norm(iTr);
    Xva = inputs_norm(iVa,:);  Yva = target_norm(iVa);
    Xte = inputs_norm(iTe,:);  Yte = target_norm(iTe);

    % Per-run thresholds from TRAINING data only (no test leakage)
    Ytr_orig = Ytr * (t_max - t_min) + t_min;
    lo = prctile(Ytr_orig, 33);
    hi = prctile(Ytr_orig, 66);

    % Build FIS via subtractive clustering (compact, data-driven)
    fis = genfis2(Xtr, Ytr, clusterRad);
    nRules = length(fis.rule);
    fprintf('  Run %2d: %d rules', run, nRules);

    % Train with validation-based early stopping via chkFis
    [trainedFis, ~, ~, chkFis] = anfis([Xtr Ytr], fis, epochs, ...
        [NaN NaN NaN NaN], [Xva Yva], stepSize);

    % Pick the better of trainedFis vs chkFis on validation data
    Yva_chk = evalfis(chkFis, Xva);
    Yva_tr  = evalfis(trainedFis, Xva);
    errC = sqrt(mean((Yva_chk - Yva).^2));
    errT = sqrt(mean((Yva_tr  - Yva).^2));
    finalFis = chkFis;
    if errT < errC, finalFis = trainedFis; end

    Yte_pred = evalfis(finalFis, Xte);
    Yva_pred = evalfis(finalFis, Xva);
    Ytr_pred = evalfis(finalFis, Xtr);

    trainRMSE(run) = sqrt(mean((Ytr_pred - Ytr).^2));
    valRMSE(run)   = sqrt(mean((Yva_pred - Yva).^2));
    testRMSE(run)  = sqrt(mean((Yte_pred - Yte).^2));

    if valRMSE(run) < bestValRMSE
        bestValRMSE = valRMSE(run);
        bestFis = finalFis;
        bestRun = run;
    end

    % Accuracy via 3-class (low / medium / high)
    yte_o = Yte * (t_max - t_min) + t_min;
    yp_o  = Yte_pred * (t_max - t_min) + t_min;

    act = zeros(size(yte_o));
    act(yte_o <= lo) = 1;
    act(yte_o > lo & yte_o < hi) = 2;
    act(yte_o >= hi) = 3;
    prd = zeros(size(yp_o));
    prd(yp_o <= lo) = 1;
    prd(yp_o > lo & yp_o < hi) = 2;
    prd(yp_o >= hi) = 3;

    accuracy(run) = sum(prd == act) / length(act) * 100;
    fprintf('  Acc=%.1f%%  Tr=%.4f  Va=%.4f  Te=%.4f\n', ...
        accuracy(run), trainRMSE(run), valRMSE(run), testRMSE(run));
end

%% Report
fprintf('\n========== %d-RUN RESULTS ==========\n', numRuns);
fprintf('Train RMSE : %.4f ± %.4f\n', mean(trainRMSE), std(trainRMSE));
fprintf('Val   RMSE : %.4f ± %.4f\n', mean(valRMSE),   std(valRMSE));
fprintf('Test  RMSE : %.4f ± %.4f\n', mean(testRMSE),  std(testRMSE));
fprintf('Accuracy   : %.2f%% ± %.2f%%\n', mean(accuracy), std(accuracy));
fprintf('Best run   : %d  (ValRMSE = %.4f,  %d rules)\n', ...
    bestRun, bestValRMSE, length(bestFis.rule));

%% Evaluate best model on full dataset
all_pred = evalfis(bestFis, inputs_norm);
all_rmse = sqrt(mean((all_pred - target_norm).^2));
all_orig_pred = all_pred * (t_max - t_min) + t_min;
all_orig_actual = target_norm * (t_max - t_min) + t_min;
fprintf('Full RMSE: %.4f (%.4f norm)\n', ...
    sqrt(mean((all_orig_pred - all_orig_actual).^2)), all_rmse);

%% Figures (only at the end — faster than per-run plots)
figure('Name','Best ANFIS — Actual vs Predicted');
scatter(all_orig_actual, all_orig_pred, 8, 'filled');
hold on;
plot(xlim, xlim, 'r--', 'LineWidth', 1.5);
xlabel('Actual Benzene (\mu g/m^3)');
ylabel('Predicted Benzene (\mu g/m^3)');
title(sprintf('Best ANFIS  |  RMSE = %.4f \\mu g/m^3', ...
    sqrt(mean((all_orig_pred - all_orig_actual).^2))));
grid on; axis equal tight;

figure('Name','FIS Structure');
plotfis(bestFis);

figure('Name','MFs of Input 1');
plotmf(bestFis, 'input', 1);

fprintf('\nElapsed: %.2f s\n', toc);