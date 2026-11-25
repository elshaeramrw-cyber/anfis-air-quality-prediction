clear; clc; close all;
rng(42);
tic;

% ============================================
% 1. ROBUST DATA LOADING (fixed European decimal)
% ============================================
filename = 'AirQualityUCI.csv';

rawText = readtable(filename, 'FileType', 'text', 'Delimiter', ';', ...
    'ReadVariableNames', false);

nRows = height(rawText);
nCols = width(rawText);
data = zeros(nRows, nCols);

fprintf('Parsing %d columns...\n', nCols);
for j = 1:nCols
    col = rawText{:, j};
    if iscell(col)
        col = strrep(col, ',', '.');
        colIdx = strcmp(col, '-200');
        col(colIdx) = {'NaN'};
        data(:, j) = str2double(col);
    elseif isnumeric(col) || islogical(col)
        data(:, j) = double(col);
        else
        % datetime or duration → convert to serial
        data(:, j) = datenum(col);
    end
end

colNames = {'Date','Time','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT', ...
            'PT08_S2_NMHC','NOx_GT','PT08_S3_NOx','NO2_GT','PT08_S4_NO2', ...
            'PT08_S5_O3','T','RH','AH'};

fprintf('Loaded %d rows x %d cols\n', nRows, nCols);
fprintf('Missing values per column:\n');
for j = 1:nCols
    nMiss = sum(isnan(data(:,j)));
    pct = nMiss / nRows * 100;
    marker = ' ';
    if pct > 30, marker = '>'; end
    fprintf('  %s %s: %d (%.1f%%)%s\n', marker, colNames{j}, nMiss, pct, ...
        repmat(' <-- DROPPED', 1, pct > 30));
end

% ============================================
% 2. DROP BAD COLUMNS & INTERPOLATE
% ============================================
validCols = true(1, nCols);
for j = 1:nCols
    pct = sum(isnan(data(:,j))) / nRows * 100;
    if pct > 30
        validCols(j) = false;
    end
end

featureCols = find(validCols);
featureCols = setdiff(featureCols, [1, 2, 6]);
nFeat = length(featureCols);
fprintf('\nUsing %d feature columns + target\n', nFeat);

Xraw = data(:, featureCols);
Yraw = data(:, 6);

for j = 1:nFeat
    Xraw(:,j) = fillmissing(Xraw(:,j), 'linear');
end
Yraw = fillmissing(Yraw, 'linear');

valid = all(~isnan(Xraw),2) & ~isnan(Yraw);
Xraw = Xraw(valid,:);
Yraw = Yraw(valid);
nSamples = size(Xraw, 1);
fprintf('After cleanup: %d samples\n', nSamples);

% ============================================
% 3. TUNE HYPERPARAMETERS ON VALIDATION SET
% ============================================
idx = randperm(nSamples);
nTr = round(0.7 * nSamples);
nVa = round(0.15 * nSamples);
iTr = idx(1:nTr);
iVa = idx(nTr+1:nTr+nVa);
iTe = idx(nTr+nVa+1:end);

% Grid search: try different feature counts and cluster radii
featCounts = [5, nFeat];
radii = [0.4, 0.5, 0.6];
nTuneRuns = 2;

corrs = zeros(nFeat,1);
for j = 1:nFeat
    r = corrcoef(Xraw(:,j), Yraw);
    corrs(j) = abs(r(1,2));
end
[~, featRank] = sort(corrs, 'descend');

fprintf('\n========== HYPERPARAMETER TUNING ==========\n');
fprintf('Feature correlations:\n');
for j = 1:nFeat
    fprintf('  %s: %.4f\n', colNames{featureCols(featRank(j))}, corrs(featRank(j)));
end

bestConfig = [];
bestValRMSE = inf;

Xtr_tune = Xraw(iTr,:);
Ytr_tune = Yraw(iTr);
Xva_tune = Xraw(iVa,:);
Yva_tune = Yraw(iVa);

trMin = min(Xtr_tune); trMax = max(Xtr_tune);
Xtr_norm = (Xtr_tune - trMin) ./ (trMax - trMin);
Xva_norm = (Xva_tune - trMin) ./ (trMax - trMin);
tMinTr = min(Ytr_tune); tMaxTr = max(Ytr_tune);
Ytr_norm = (Ytr_tune - tMinTr) ./ (tMaxTr - tMinTr);
Yva_norm = (Yva_tune - tMinTr) ./ (tMaxTr - tMinTr);

for nF = featCounts
    nF_actual = min(nF, nFeat);
    selFeats = featRank(1:nF_actual);

    for r = radii
        valErrs = zeros(nTuneRuns, 1);
        nRules = 0;

        for run = 1:nTuneRuns
            fis = genfis2(Xtr_norm(:, selFeats), Ytr_norm, r);
            nRules = length(fis.rule);
            
            [~, ~, ~, chkFis] = anfis([Xtr_norm(:, selFeats) Ytr_norm], fis, ...
            200, [0 0 0], [Xva_norm(:, selFeats) Yva_norm], 0.02);

            Yv = evalfis(chkFis, Xva_norm(:, selFeats));
            valErrs(run) = sqrt(mean((Yv - Yva_norm).^2));
        end

        meanVal = mean(valErrs);
        fprintf('  nFeat=%d (%d actual), radius=%.1f, rules=%d, ValRMSE=%.4f\n', ...
            nF, nF_actual, r, nRules, meanVal);

        if meanVal < bestValRMSE
            bestValRMSE = meanVal;
            bestConfig = [nF_actual, r];
        end
    end
end

nFeatBest = bestConfig(1);
radiusBest = bestConfig(2);
fprintf('\nBest config: %d features, radius=%.1f (ValRMSE=%.4f)\n', ...
    nFeatBest, radiusBest, bestValRMSE);

% ============================================
% 4. FULL EVALUATION WITH BEST CONFIG
% ============================================
selFeats = featRank(1:nFeatBest);
fprintf('\nSelected features for evaluation:\n');
for j = 1:nFeatBest
    fprintf('  %s (corr=%.4f)\n', colNames{featureCols(selFeats(j))}, corrs(selFeats(j)));
end

inputs = Xraw(:, selFeats);
target = Yraw;

inMin = min(inputs);  inMax = max(inputs);
inputsNorm = (inputs - inMin) ./ (inMax - inMin);
tMin = min(target);   tMax = max(target);
targetNorm = (target - tMin) ./ (tMax - tMin);

numRuns = 10;
epochs = 200;

trainRMSE = zeros(numRuns,1);
valRMSE   = zeros(numRuns,1);
testRMSE  = zeros(numRuns,1);
accuracy  = zeros(numRuns,1);
nRulesAll = zeros(numRuns,1);

bestValRMSE_run = inf;
bestFis = [];
bestRun = 0;

fprintf('\n========== %d-RUN EVALUATION (best config) ==========\n', numRuns);

for run = 1:numRuns
    idx = randperm(nSamples);
    iTr = idx(1:nTr);
    iVa = idx(nTr+1:nTr+nVa);
    iTe = idx(nTr+nVa+1:end);

    Xtr = inputsNorm(iTr,:);  Ytr = targetNorm(iTr);
    Xva = inputsNorm(iVa,:);  Yva = targetNorm(iVa);
    Xte = inputsNorm(iTe,:);  Yte = targetNorm(iTe);

    Ytr_orig = Ytr * (tMax - tMin) + tMin;
    lo = prctile(Ytr_orig, 33);
    hi = prctile(Ytr_orig, 66);

    fis = genfis2(Xtr, Ytr, radiusBest);
    nRulesAll(run) = length(fis.rule);

    [trainedFis, ~, ~, chkFis] = anfis([Xtr Ytr], fis, epochs, ...
        [NaN NaN NaN NaN], [Xva Yva], 0.02);

    valC = sqrt(mean((evalfis(chkFis, Xva) - Yva).^2));
    valT = sqrt(mean((evalfis(trainedFis, Xva) - Yva).^2));
    finalFis = chkFis;
    if valT < valC, finalFis = trainedFis; end

    Ytr_p = evalfis(finalFis, Xtr);
    Yva_p = evalfis(finalFis, Xva);
    Yte_p = evalfis(finalFis, Xte);

    trainRMSE(run) = sqrt(mean((Ytr_p - Ytr).^2));
    valRMSE(run)   = sqrt(mean((Yva_p - Yva).^2));
    testRMSE(run)  = sqrt(mean((Yte_p - Yte).^2));

    if valRMSE(run) < bestValRMSE_run
        bestValRMSE_run = valRMSE(run);
        bestFis = finalFis;
        bestRun = run;
    end

    Yte_o = Yte * (tMax - tMin) + tMin;
    Yte_po = Yte_p * (tMax - tMin) + tMin;

    act = zeros(size(Yte_o));
    act(Yte_o <= lo) = 1;
    act(Yte_o > lo & Yte_o < hi) = 2;
    act(Yte_o >= hi) = 3;
    prd = zeros(size(Yte_po));
    prd(Yte_po <= lo) = 1;
    prd(Yte_po > lo & Yte_po < hi) = 2;
    prd(Yte_po >= hi) = 3;

    accuracy(run) = sum(prd == act) / length(act) * 100;
    fprintf('Run %2d: Acc=%.1f%%, Tr=%.4f, Va=%.4f, Te=%.4f, rules=%d\n', ...
        run, accuracy(run), trainRMSE(run), valRMSE(run), testRMSE(run), nRulesAll(run));
end

fprintf('\n========== RESULTS ==========\n');
fprintf('Test RMSE : %.4f +/- %.4f\n', mean(testRMSE), std(testRMSE));
fprintf('Accuracy  : %.2f%% +/- %.2f%%\n', mean(accuracy), std(accuracy));
fprintf('Rules     : %.0f +/- %.0f\n', mean(nRulesAll), std(nRulesAll));
fprintf('Best run  : %d\n', bestRun);

allPred = evalfis(bestFis, inputsNorm);
allPredOrig = allPred * (tMax - tMin) + tMin;
fprintf('\nFull-dataset RMSE: %.4f (%.4f norm)\n', ...
    sqrt(mean((allPredOrig - target).^2)), sqrt(mean((allPred - targetNorm).^2)));

% ============================================
% 5. VISUALIZATION
% ============================================
figure('Name','Best ANFIS - Actual vs Predicted','Position',[100 100 600 500]);
scatter(target, allPredOrig, 8, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(xlim, xlim, 'r--', 'LineWidth', 1.5);
xlabel('Actual Benzene (\mug/m^3)');
ylabel('Predicted Benzene (\mug/m^3)');
rmseFinal = sqrt(mean((allPredOrig - target).^2));
title(sprintf('Best ANFIS: RMSE = %.4f \\mug/m^3, Accuracy = %.1f%%', ...
    rmseFinal, mean(accuracy)));
grid on; axis equal tight;

figure('Name','FIS Structure');
plotfis(bestFis);

if nFeatBest >= 1
    figure('Name','Membership Functions');
    plotmf(bestFis, 'input', 1);
end

loFinal = prctile(target, 33);
hiFinal = prctile(target, 66);
actFinal = zeros(size(target));
actFinal(target <= loFinal) = 1;
actFinal(target > loFinal & target < hiFinal) = 2;
actFinal(target >= hiFinal) = 3;
prdFinal = zeros(size(allPredOrig));
prdFinal(allPredOrig <= loFinal) = 1;
prdFinal(allPredOrig > loFinal & allPredOrig < hiFinal) = 2;
prdFinal(allPredOrig >= hiFinal) = 3;

figure('Name','Confusion Matrix');
cm = confusionmat(actFinal, prdFinal);
h = heatmap(cm, ...
    'XDisplayLabels', {'Low','Medium','High'}, ...
    'YDisplayLabels', {'Low','Medium','High'});
h.Title = sprintf('Confusion Matrix (Accuracy: %.1f%%)', ...
    sum(diag(cm))/sum(cm(:))*100);
h.XLabel = 'Predicted Class';
h.YLabel = 'Actual Class';

fprintf('\nElapsed: %.2f s\n', toc);