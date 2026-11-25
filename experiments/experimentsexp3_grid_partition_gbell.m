clear; clc; close all;
rng(42);

filename = 'AirQualityUCI.csv';

data_table = readtable(filename, 'FileType', 'text', 'Delimiter', ';', ...
    'ReadVariableNames', false, 'TreatAsMissing', '-200');

fprintf('Original rows: %d\n', height(data_table));
data_table = data_table(:, 1:15);

varNames = {'Date','Time','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT', ...
            'PT08_S2_NMHC','NOx_GT','PT08_S3_NOx','NO2_GT','PT08_S4_NO2', ...
            'PT08_S5_O3','T','RH','AH'};
data_table.Properties.VariableNames = varNames;

numericVars = setdiff(varNames, {'Date','Time'});
numericVars = numericVars(:)';

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

remaining_missing = sum(any(ismissing(data_table{:, numericVars}), 2));
fprintf('Rows with missing after interpolation: %d\n', remaining_missing);

%% Inputs and target
inputs = data_table{:, {'PT08_S1_CO','PT08_S2_NMHC','PT08_S3_NOx','PT08_S4_NO2','PT08_S5_O3'}};
target = data_table{:, 'C6H6_GT'};

% Smooth target
target = movmean(target, 5);

% Global thresholds for classification
global_low = prctile(target, 33);
global_high = prctile(target, 66);
fprintf('Global thresholds: low=%.2f, high=%.2f µg/m³\n', global_low, global_high);

% Remove edge NaNs (if any)
valid = all(~isnan(inputs),2) & ~isnan(target);
if sum(~valid) > 0
    fprintf('Removing %d rows where interpolation failed.\n', sum(~valid));
    inputs = inputs(valid,:);
    target = target(valid);
end
fprintf('Final dataset size: %d rows\n', size(inputs,1));

%% Normalize
min_input = min(inputs);
max_input = max(inputs);
inputs_norm = (inputs - min_input) ./ (max_input - min_input);
min_target = min(target);
max_target = max(target);
target_norm = (target - min_target) ./ (max_target - min_target);

%% Parameters
numRuns = 10;
trainRatio = 0.7;
valRatio = 0.15;
numMFs = 3;
mfType = 'gbellmf';   % try 'gbellmf' instead of 'gaussmf'
epochs = 300;
stepSize = 0.02;

options = genfisOptions('GridPartition');
options.NumMembershipFunctions = numMFs;
options.InputMembershipFunctionType = mfType;

trainRMSE = zeros(numRuns,1);
valRMSE   = zeros(numRuns,1);
testRMSE  = zeros(numRuns,1);
accuracy  = zeros(numRuns,1);
final_fis = [];

fprintf('Starting %d runs with 70/15/15 split...\n', numRuns);

for run = 1:numRuns
    nSamples = size(inputs_norm,1);
    idx = randperm(nSamples);
    nTrain = round(trainRatio * nSamples);
    nVal   = round(valRatio * nSamples);
    
    idx_train = idx(1:nTrain);
    idx_val   = idx(nTrain+1 : nTrain+nVal);
    idx_test  = idx(nTrain+nVal+1 : end);
    
    X_train = inputs_norm(idx_train, :);
    Y_train = target_norm(idx_train);
    X_val   = inputs_norm(idx_val, :);
    Y_val   = target_norm(idx_val);
    X_test  = inputs_norm(idx_test, :);
    Y_test  = target_norm(idx_test);
    
    fis = genfis(X_train, Y_train, options);
    [trainedFis, ~, ~, ~] = anfis([X_train Y_train], fis, epochs, ...
        [NaN NaN NaN NaN], [X_val Y_val], stepSize);
    
    Y_train_pred = evalfis(trainedFis, X_train);
    Y_val_pred   = evalfis(trainedFis, X_val);
    Y_test_pred  = evalfis(trainedFis, X_test);
    
    trainRMSE(run) = sqrt(mean((Y_train_pred - Y_train).^2));
    valRMSE(run)   = sqrt(mean((Y_val_pred - Y_val).^2));
    testRMSE(run)  = sqrt(mean((Y_test_pred - Y_test).^2));
    
    Y_test_orig = Y_test * (max_target - min_target) + min_target;
    Y_test_pred_orig = Y_test_pred * (max_target - min_target) + min_target;
    
    % Use global thresholds
    low_thresh = global_low;
    high_thresh = global_high;
    
    actual_class = zeros(size(Y_test_orig));
    pred_class = zeros(size(Y_test_orig));
    actual_class(Y_test_orig <= low_thresh) = 1;
    actual_class(Y_test_orig > low_thresh & Y_test_orig < high_thresh) = 2;
    actual_class(Y_test_orig >= high_thresh) = 3;
    pred_class(Y_test_pred_orig <= low_thresh) = 1;
    pred_class(Y_test_pred_orig > low_thresh & Y_test_pred_orig < high_thresh) = 2;
    pred_class(Y_test_pred_orig >= high_thresh) = 3;
    
    accuracy(run) = sum(pred_class == actual_class) / length(actual_class) * 100;
    fprintf('Run %2d: Accuracy = %.2f%%\n', run, accuracy(run));
    
    figure('Visible', 'off');
    scatter(Y_test_orig, Y_test_pred_orig, 10, 'filled');
    hold on;
    plot([min(Y_test_orig), max(Y_test_orig)], [min(Y_test_orig), max(Y_test_orig)], 'r--', 'LineWidth', 1.5);
    xlabel('Actual Benzene (µg/m³)');
    ylabel('Predicted Benzene (µg/m³)');
    title(sprintf('Run %d: Actual vs Predicted (Test Set)', run));
    grid on;
    saveas(gcf, sprintf('scatter_run_%02d.png', run));
    close(gcf);
    
    if run == numRuns
        final_fis = trainedFis;
    end
    
    fprintf('Run %2d: TrainRMSE=%.4f, ValRMSE=%.4f, TestRMSE=%.4f\n', ...
        run, trainRMSE(run), valRMSE(run), testRMSE(run));
end

%% Results
fprintf('\n========== RESULTS OVER %d RUNS ==========\n', numRuns);
fprintf('Training RMSE   : %.4f ± %.4f\n', mean(trainRMSE), std(trainRMSE));
fprintf('Validation RMSE : %.4f ± %.4f\n', mean(valRMSE), std(valRMSE));
fprintf('Test RMSE       : %.4f ± %.4f\n', mean(testRMSE), std(testRMSE));
fprintf('Average accuracy: %.2f%% ± %.2f%%\n', mean(accuracy), std(accuracy));

%% Visualize final FIS
figure('Name', 'Final ANFIS Structure');
plotfis(final_fis);
figure('Name', 'Membership Functions of First Input');
plotmf(final_fis, 'input', 1);
all_outputs = target_norm;
all_pred = evalfis(final_fis, inputs_norm);
figure('Name', 'Actual vs Predicted (Normalized)');
scatter(all_outputs, all_pred, 10, 'filled');
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Actual Benzene (normalized)');
ylabel('Predicted Benzene (normalized)');
title(sprintf('ANFIS: RMSE = %.4f (normalized)', mean(testRMSE)));
grid on;