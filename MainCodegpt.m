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
ALL = [ratioWO, WOpainpre, genotype ROIs];
%ALL(ratioWO>0.9999, :) = NaN;
ALL = normvalues(ALL);
%ALL = standardizeValues(ALL);
varname = 'Normalised Improvement WO';

% Remove rows with NaN values
ALL(isnan(ALL(:, 1)), :) = [];
input = ALL(:, 2:end);
target = ALL(:, 1);

%%

% Set up RFE options
options = statset('UseParallel', true);

% Initialize result containers
mseValues = zeros(size(target));
numSelectedFeatures = 15;  % Adjust as needed

% Initialize result containers
predictions1 = zeros(1, numel(target));
predictedTarget = zeros(1, numel(target));

numFolds =numel(predictions1);

% Leave-one-out cross-validation loop
for fold = randperm(numFolds)
    % Calculate the indices for training and testing sets
    testIndices = round((fold - 1) * numel(target) / numFolds + 1 : fold * numel(target) / numFolds);
    trainIndices = setdiff(1:numel(target), testIndices);

    % Create training and testing sets
    targetTrain = target(trainIndices);
    targetTest = target(testIndices);
    
    inputTrain = input(trainIndices, :);
    inputTest = input(testIndices, :);
   
    % Feature selection using RFE with SVM
    mdl = fitrsvm(inputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true, 'BoxConstraint', 1);
    featureWeights = abs(mdl.Beta);
    [~, sortedIndices] = sort(featureWeights, 'descend');
    selectedFeaturesidx = sortedIndices(1:numSelectedFeatures);

    % Train the final model using selected features
    selectedInputTrain = inputTrain(:, selectedFeaturesidx);
    selectedInputTest = inputTest(:, selectedFeaturesidx);
    
    
    finalModel = fitrsvm(selectedInputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);

    % Make predictions on the test data
    predictedTarget(testIndices) = predict(finalModel, selectedInputTest);

    % Evaluate the performance (you can replace this with your own evaluation metrics)
    fprintf('Progress: %.2f%%\n', 100 * testIndices/ numel(target));
    
    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain(:, 1), targetTrain);
    predictions1(testIndices) = predict(mdl1, inputTest(1));
end

%%
figure(2)
subplot(1,3,1)
[rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictions1', 30, 'b');
title({"Model: [PainPre]", sprintf("vs %s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0.1,1.5])
hold off
subplot(1,3,2)
[rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictedTarget', 30, 'b');
title({"Model: [PainPre, ROIs, geno]", sprintf("vs %s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0.1,1.5])
hold off

subplot(1,3,3)
[rho2, p2] = PlotSimpleCorrelationWithRegression(targetTrain, predict(finalModel, selectedInputTrain), 30, [0.5 0.5 0.5]);
title({"Train: [PainPre, ROIs, geno]", sprintf("vs %s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0.1,1.5])
hold off


%%
% Feature selection criterion function for RFE
function crit = critfun(input, target)
    mdl = fitrsvm(input, target, 'KernelFunction', 'linear', 'Standardize', true);
    crit = resubLoss(mdl, 'LossFun', 'mse');
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
