clear; clc; close all;
rng(42);
readtable AirQualityUCI.csv;
filename ='AirQualityUCI.csv';

data_table = readtable(filename, 'FileType', 'text', 'Delimiter', ';', ...
    'ReadVariableNames', false, 'TreatAsMissing', '-200');

fprintf('Original rows: %d\n', height(data_table));


data_table = data_table(:, 1:15);


varNames = {'Date','Time','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT', ...
            'PT08_S2_NMHC','NOx_GT','PT08_S3_NOx','NO2_GT','PT08_S4_NO2', ...
            'PT08_S5_O3','T','RH','AH'};
data_table.Properties.VariableNames = varNames;


numericVars = setdiff(varNames, {'Date','Time'});
numericVars = numericVars(:)';   % ensure row cell array

% Convert each numeric column from strings (with commas as decimals) to double
for i = 1:length(numericVars)
    colName = numericVars{i};
    colData = data_table.(colName);
    if iscell(colData)
        % Replace comma with dot and convert to double
        colData = strrep(colData, ',', '.');
        data_table.(colName) = str2double(colData);
    else
        data_table.(colName) = double(colData);
    end
    data_table.(colName) = fillmissing(data_table.(colName), 'linear');
end

% Verify no missing values remain
remaining_missing = sum(any(ismissing(data_table{:, numericVars}), 2));
fprintf('Rows with missing after interpolation: %d\n', remaining_missing);

%% 4. Extract inputs and target
inputs = data_table{:, {'PT08_S1_CO','PT08_S2_NMHC','PT08_S3_NOx','PT08_S4_NO2','PT08_S5_O3'}};
target = data_table{:, 'C6H6_GT'};

% If any NaN still exist (e.g., at the very beginning or end where interpolation fails),
% remove only those extreme rows (usually very few, < 10)
valid = all(~isnan(inputs),2) & ~isnan(target);
if sum(~valid) > 0
    fprintf('Removing %d rows where interpolation failed (likely at edges).\n', sum(~valid));
    inputs = inputs(valid,:);
    target = target(valid);
end

fprintf('Final dataset size: %d rows\n', size(inputs,1));

%% 3. Normalize data to [0,1] (min-max)
min_input = min(inputs);
max_input = max(inputs);
inputs_norm = (inputs - min_input) ./ (max_input - min_input);

min_target = min(target);
max_target = max(target);
target_norm = (target - min_target) ./ (max_target - min_target);

%% 4. Parameters for experiments
numRuns = 10;
trainRatio = 0.7;   % 70% training
valRatio   = 0.15;  % 15% validation
testRatio  = 0.15;  % 15% testing

% ANFIS options
numMFs = 3;          % number of membership functions per input
mfType = 'gaussmf';  % Gaussian MF
epochs = 250;
stepSize = 0.02;     % initial step size for backpropagation (optional)
optimMethod = 1;     % 1: hybrid (default)
options = genfisOptions('GridPartition');
options.NumMembershipFunctions = numMFs;   % e.g., 3
options.InputMembershipFunctionType = mfType; % e.g., 'gbellmf'

% Store results
trainRMSE = zeros(numRuns,1);
valRMSE   = zeros(numRuns,1);
testRMSE  = zeros(numRuns,1);

% For later display of final model
final_fis = [];

%% 5. Run multiple experiments
fprintf('Starting %d runs with 70/15/15 split...\n', numRuns);

for run = 1:numRuns
    % 5.1 Randomly split data (stratification not needed for regression)
    nSamples = size(inputs_norm,1);
    idx = randperm(nSamples);

    nTrain = round(trainRatio * nSamples);
    nVal   = round(valRatio * nSamples);
    % test gets the rest
    
    idx_train = idx(1:nTrain);
    idx_val   = idx(nTrain+1 : nTrain+nVal);
    idx_test  = idx(nTrain+nVal+1 : end);
    
    X_train = inputs_norm(idx_train, :);
    Y_train = target_norm(idx_train);
    
    X_val   = inputs_norm(idx_val, :);
    Y_val   = target_norm(idx_val);
    
    X_test  = inputs_norm(idx_test, :);
    Y_test  = target_norm(idx_test);
    
    % 5.2 Generate initial FIS using grid partition
    fis = genfis(X_train, Y_train, options);
    
    % 5.3 Train ANFIS (use training + validation to avoid overfitting)
    % We pass X_train, Y_train and also validation data
    [trainedFis, trainError, ~, valError] = anfis([X_train Y_train], fis, epochs, ...
        [NaN NaN NaN NaN], [X_val Y_val], stepSize);
    
    % 5.4 Evaluate on train, validation, test
    Y_train_pred = evalfis(trainedFis, X_train);
    Y_val_pred   = evalfis(trainedFis, X_val);
    Y_test_pred  = evalfis(trainedFis, X_test);
    
    % 5.5 Compute RMSE (normalized scale)
    trainRMSE(run) = sqrt(mean((Y_train_pred - Y_train).^2));
    valRMSE(run)   = sqrt(mean((Y_val_pred - Y_val).^2));
    testRMSE(run)  = sqrt(mean((Y_test_pred - Y_test).^2));
    
    Y_test_orig = Y_test * (max_target - min_target) + min_target;
    Y_test_pred_orig = Y_test_pred * (max_target - min_target) + min_target;

    low_thresh = prctile(Y_test_orig, 33);
    high_thresh = prctile(Y_test_orig, 66);


    actual_class = zeros(size(Y_test_orig));
    pred_class = zeros(size(Y_test_orig));

    actual_class(Y_test_orig <= low_thresh) = 1;      % Low risk
    actual_class(Y_test_orig > low_thresh & Y_test_orig < high_thresh) = 2;  % Medium
    actual_class(Y_test_orig >= high_thresh) = 3;     % High risk

    pred_class(Y_test_pred_orig <= low_thresh) = 1;
    pred_class(Y_test_pred_orig > low_thresh & Y_test_pred_orig < high_thresh) = 2;
    pred_class(Y_test_pred_orig >= high_thresh) = 3;
    
    accuracy(run) = sum(pred_class == actual_class) / length(actual_class) * 100;
    fprintf('Run %2d: Accuracy = %.2f%%\n', run, accuracy(run));

    figure('Visible', 'on');  % 'off' to avoid pop-ups; set 'on' if you want to see them
    scatter(Y_test_orig, Y_test_pred_orig, 10, 'filled');
    hold on;
    plot([min(Y_test_orig), max(Y_test_orig)], [min(Y_test_orig), max(Y_test_orig)], 'r--', 'LineWidth', 1.5);
    xlabel('Actual Benzene (µg/m³)');
    ylabel('Predicted Benzene (µg/m³)');
    title(sprintf('Run %d: Actual vs Predicted (Test Set)', run));
    grid on;
    saveas(gcf, sprintf('scatter_run_%02d.png', run));
    close(gcf);
    
    

    % Store the last trained FIS for visualization
    if run == numRuns
        final_fis = trainedFis;
    end
    
    fprintf('Run %2d: TrainRMSE=%.4f, ValRMSE=%.4f, TestRMSE=%.4f\n', ...
        run, trainRMSE(run), valRMSE(run), testRMSE(run));
end

%% 6. Compute average and standard deviation over 10 runs
avg_trainRMSE = mean(trainRMSE);
std_trainRMSE = std(trainRMSE);
avg_valRMSE   = mean(valRMSE);
std_valRMSE   = std(valRMSE);
avg_testRMSE  = mean(testRMSE);
std_testRMSE  = std(testRMSE);
    
actual_class(Y_test_orig <= low_thresh) = 1;      % Low risk
actual_class(Y_test_orig > low_thresh & Y_test_orig < high_thresh) = 2;  % Medium
actual_class(Y_test_orig >= high_thresh) = 3;     % High risk

pred_class(Y_test_pred_orig <= low_thresh) = 1;
pred_class(Y_test_pred_orig > low_thresh & Y_test_pred_orig < high_thresh) = 2;
pred_class(Y_test_pred_orig >= high_thresh) = 3;

overall_avg_acc = mean(accuracy);
overall_std_acc = std(accuracy);

fprintf('\n========== RESULTS OVER %d RUNS ==========\n', numRuns);
fprintf('Training RMSE   : %.4f ± %.4f\n', avg_trainRMSE, std_trainRMSE);
fprintf('Validation RMSE : %.4f ± %.4f\n', avg_valRMSE, std_valRMSE);
fprintf('Test RMSE       : %.4f ± %.4f\n', avg_testRMSE, std_testRMSE);
fprintf('\n========== CLASSIFICATION ACCURACY OVER %d RUNS ==========\n', numRuns);
fprintf('Average accuracy: %.2f%% ± %.2f%%\n', overall_avg_acc, overall_std_acc);

%% 7. Visualize final fuzzy system
figure('Name', 'Final ANFIS Structure');
plotfis(final_fis);

figure('Name', 'Membership Functions of First Input');
plotmf(final_fis, 'input', 1);
% Use entire dataset to show general fit
all_outputs = target_norm;
all_pred = evalfis(final_fis, inputs_norm);
figure('Name', 'Actual vs Predicted (Normalized)');
scatter(all_outputs, all_pred, 10, 'filled');
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Actual Benzene (normalized)');
ylabel('Predicted Benzene (normalized)');
title(sprintf('ANFIS: RMSE = %.4f (normalized)', avg_testRMSE));
grid on;