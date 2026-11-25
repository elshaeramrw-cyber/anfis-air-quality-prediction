clear; clc; close all;
rng(42);

filename = 'AirQualityUCI.csv';

data_table = readtable(filename, 'FileType', 'text', 'Delimiter', ';', ...
    'ReadVariableNames', false, 'TreatAsMissing', '-200');

data_table = data_table(:, 1:15);

varNames = {'Date','Time','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT', ...
            'PT08_S2_NMHC','NOx_GT','PT08_S3_NOx','NO2_GT','PT08_S4_NO2', ...
            'PT08_S5_O3','T','RH','AH'};
data_table.Properties.VariableNames = varNames;

numericVars = setdiff(varNames, {'Date','Time'});
numericVars = numericVars(:)';

% Convert to numeric and interpolate
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

% Extract all sensor inputs and target
all_sensors = data_table{:, {'PT08_S1_CO','PT08_S2_NMHC','PT08_S3_NOx','PT08_S4_NO2','PT08_S5_O3'}};
target = data_table{:, 'C6H6_GT'};

% Select top 2 sensors by correlation
corr_vals = zeros(size(all_sensors,2),1);
for i = 1:size(all_sensors,2)
    corr_vals(i) = corr(all_sensors(:,i), target);
end
[~, sorted_idx] = sort(abs(corr_vals), 'descend');
inputs = all_sensors(:, sorted_idx(1:2));  % only 2 inputs

% Smooth target
target = movmean(target, 5);

% Remove any remaining NaNs
valid = all(~isnan(inputs),2) & ~isnan(target);
inputs = inputs(valid,:);
target = target(valid);

% Normalize
inputs_norm = (inputs - min(inputs)) ./ (max(inputs) - min(inputs));
target_norm = (target - min(target)) ./ (max(target) - min(target));

% Parameters
numRuns = 2;  % just 2 runs for testing
numMFs = 2;
mfType = 'gaussmf';
epochs = 50;
stepSize = 0.02;

options = genfisOptions('GridPartition');
options.NumMembershipFunctions = numMFs;
options.InputMembershipFunctionType = mfType;

trainRMSE = zeros(numRuns,1);
testRMSE = zeros(numRuns,1);
accuracy = zeros(numRuns,1);

% Global thresholds (percentiles)
global_low = prctile(target, 33);
global_high = prctile(target, 66);

fprintf('Using 2 inputs, %d MFs each → %d rules\n', numMFs, numMFs^2);

for run = 1:numRuns
    % Split
    n = size(inputs_norm,1);
    idx = randperm(n);
    nTrain = round(0.7*n);
    X_train = inputs_norm(idx(1:nTrain), :);
    Y_train = target_norm(idx(1:nTrain));
    X_test = inputs_norm(idx(nTrain+1:end), :);
    Y_test = target_norm(idx(nTrain+1:end));
    
    % Generate FIS and train
    fis = genfis(X_train, Y_train, options);
    [trainedFis, ~] = anfis([X_train Y_train], fis, epochs, [NaN NaN NaN NaN], [], stepSize);
    
    % Predict
    Y_pred = evalfis(trainedFis, X_test);
    testRMSE(run) = sqrt(mean((Y_pred - Y_test).^2));
    
    % Denormalize
    Y_test_orig = Y_test * (max(target)-min(target)) + min(target);
    Y_pred_orig = Y_pred * (max(target)-min(target)) + min(target);
    
    % Classification accuracy
    actual = zeros(size(Y_test_orig));
    pred = zeros(size(Y_test_orig));
    actual(Y_test_orig <= global_low) = 1;
    actual(Y_test_orig > global_low & Y_test_orig < global_high) = 2;
    actual(Y_test_orig >= global_high) = 3;
    pred(Y_pred_orig <= global_low) = 1;
    pred(Y_pred_orig > global_low & Y_pred_orig < global_high) = 2;
    pred(Y_pred_orig >= global_high) = 3;
    accuracy(run) = sum(pred == actual)/length(actual)*100;
    
    fprintf('Run %d: TestRMSE=%.4f, Accuracy=%.2f%%\n', run, testRMSE(run), accuracy(run));
end

fprintf('\nAverage test RMSE: %.4f ± %.4f\n', mean(testRMSE), std(testRMSE));
fprintf('Average accuracy: %.2f%% ± %.2f%%\n', mean(accuracy), std(accuracy));