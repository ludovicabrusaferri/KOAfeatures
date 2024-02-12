% Clear workspace and command window
clear all;
clc;
rng(42);

% Set the data file path
dataFilePath = '/Users/e410377/Desktop/KOAfeatures/FINALWOMAC.xlsx';
addpath(genpath('/Users/e410377/Desktop/PETAnalysisPaper/utility'));

% Import data from Excel file
data = importdata(dataFilePath);

% Extract numeric data and corresponding titles
numericData = data.data;
numericTitles = data.textdata(1, 2:end);

% Extract relevant variables
age = numericData(:, strcmp(numericTitles, 'Age (pre)'));
genotype = numericData(:, strcmp(numericTitles, 'Genotype (1=GG)'));
TKApainpre = numericData(:, strcmp(numericTitles, 'TKA pain pre'));
TKApainpost = numericData(:, strcmp(numericTitles, 'TKA pain post'));
WOpainpre = numericData(:, strcmp(numericTitles, 'Pre-TKA,WOMAC (pain)'));
WOpainpost = numericData(:, strcmp(numericTitles, '1yr POST-TKA, Womac (pain)'));
Promispre = numericData(:, strcmp(numericTitles, 'PRE-TKA, Promis (pain intensity)'));
WOphyspre = numericData(:, strcmp(numericTitles, 'Pre-TKA,WOMAC (physical function)'));

% Small epsilon value for numerical stability
eps = 0.00000000001;

% Calculate "RawChange" and corresponding ratio for TKA
rawChangeTKA = TKApainpost - TKApainpre;
ratioTKA = (TKApainpre - TKApainpost) ./ (TKApainpost + TKApainpre + eps);

% Calculate "RawChange" and corresponding ratio for WOMAC
rawChangeWO = WOpainpost - WOpainpre;
ratioWO = (WOpainpre - WOpainpost) ./ (WOpainpost + WOpainpre + eps);

% Extract SC and CC variables for further analysis
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);

% Fit linear models for TKA and WOMAC
mdlTKA = fitglm(rawChangeTKA, TKApainpre);
rawChangeTKAadj = mdlTKA.Residuals.Raw;

mdlWO = fitglm(rawChangeWO, WOpainpre);
rawChangeWOadj = mdlWO.Residuals.Raw;

%% PREDICTIONS

% Combine relevant features for prediction
ALL = [ratioWO, WOpainpre, ROIs, genotype];
%sALL(ratioWO>0.9999, :) = NaN;
ALL = normvalues(ALL);

varname = 'Normalised Improvement WO';

% Remove rows with NaN values
ALL(isnan(ALL(:, 1)), :) = [];
input = ALL(:, 2:end);
target = ALL(:, 1);

% Set options for feature selection
options = statset('UseParallel', true);
method = "SVM"; % Specify the method
numSelectedFeatures = 15; % Choose the desired number of features
linearmodel = 0;

% Initialize result containers
predictions1 = zeros(1, numel(target));
predictions2 = zeros(1, numel(target));
selectedFeaturesSTORE = zeros(size(input, 2), 1);
STORE = [];

for i = 1:numel(target)
    % Create training and testing sets
    targetTrain = target;
    targetTrain(i) = [];
    targetTest = target(i);
    
    inputTrain = input;
    inputTrain(i, :) = [];
    inputTest = input(i, :);

    % Feature selection based on the chosen method
    if strcmp(method, 'SequentialsVariable')
        % Optimize for the optimal number of features
        selectedFeaturesidx = sequentialfs(@critfun, inputTrain, targetTrain, 'cv', 'none', 'options', options);
    elseif strcmp(method, 'SequentialsFixed')
        % Choose a fixed number of features
        selectedFeaturesidx = sequentialfs(@critfun, inputTrain, targetTrain, 'cv', 'none', 'options', options, 'NFeatures', numSelectedFeatures);
    elseif strcmp(method, 'Lasso')
        % L1 regularization (LASSO) for feature selection
        [B, FitInfo] = lasso(inputTrain, targetTrain, 'CV', 20); % You can adjust 'CV' as needed
        idxLambdaMinMSE = FitInfo.IndexMinMSE;
        selectedFeaturesidx = B(:, idxLambdaMinMSE) ~= 0;
    elseif strcmp(method, 'RandomForest')
        % Train Random Forest with OOBPermuteVarDeltaError
        numTrees = 30; % You can adjust the number of trees
        baggedTree = TreeBagger(numTrees, inputTrain, targetTrain, 'Method', 'regression', 'OOBPredictorImportance', 'on');
        importance = baggedTree.OOBPermutedVarDeltaError;
        [~, sortedIdx] = sort(importance, 'descend');
        selectedFeaturesidx = sortedIdx(1:numSelectedFeatures);
    elseif strcmp(method, "SVM")
        % SVM-based feature selection for regression
        svmModel = fitrsvm(inputTrain, targetTrain, 'Standardize', true, 'KernelFunction', 'linear');
        featureWeights = abs(svmModel.Beta);
        [~, sortedIndices] = sort(featureWeights, 'descend');
        selectedFeaturesidx = sortedIndices(1:numSelectedFeatures);
    end

    inputTrainSelected = inputTrain(:, selectedFeaturesidx);
    inputTestSelected = inputTest(:, selectedFeaturesidx);

    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain(:, 1), targetTrain);
    predictions1(i) = predict(mdl1, inputTest(1));
    
    % Fit linear model or SVM with selected predictors for combined vs target
    if linearmodel
        mdlCombined = fitlm(inputTrainSelected, targetTrain);
        %mdlCombined = fitrlinear(inputTrainSelected, targetTrain, 'Regularization', 'ridge', 'Lambda', 'auto');

    else
        %mdlCombined = fitrsvm(inputTrainSelected, targetTrain, 'Standardize', true,  'KernelFunction', 'linear');
        mdlCombined = fitrsvm(inputTrainSelected, targetTrain, 'KernelFunction','linear','KernelScale','auto', 'Standardize',true);


    end
    
    predictions2(i) = predict(mdlCombined, inputTestSelected);
    
    % Store the selected features for the current fold
    selectedFeaturesSTORE(selectedFeaturesidx) = 1;
    STORE = [STORE, selectedFeaturesSTORE];
    % Reset selectedFeaturesSTORE for the next fold
    selectedFeaturesSTORE = zeros(size(input, 2), 1);
    fprintf('Progress: %.2f%%\n', 100 * i / numel(target));
end

%% PLOT RESULTS

% Plot correlations and regression lines for input vs target and combined vs target
figure(1)
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(target, predictions1', 30, 'b');
title({"Model: PainPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0,1.5])
hold off

subplot(1, 2, 2);
[rho2, p2] = PlotSimpleCorrelationWithRegression(target, predictions2', 30, 'b');
title({"Model: [PainPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0,1.5])
hold off

% Sum the frequencies across folds for selected features
freq = sum(STORE, 2);

% Plot histogram of selected features frequencies
figure(2);
bar(freq);
xlabel('Feature Index');
ylabel('Frequency across Folds');
title('Selected Features Frequencies over Folds');
set(gcf, 'Color', 'w');
set(gca, 'FontSize', 25);

%% CLUSTER

% ... (remaining code for clustering, if any)

%%
function crit = critfun(x, y)
    mdl = fitlm(x, y);
    crit = mdl.RMSE;
end

function out = normvalues(input)
    [rows, cols] = size(input);
    out = zeros(rows, cols);

    for col = 1:cols
        % Normalize each column separately
        colData = input(:, col);
        out(:, col) = (colData - min(colData)) ./ (max(colData) - min(colData));
    end
end
