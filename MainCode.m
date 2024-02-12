% Clear workspace and command window
clear all;
clc;

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

% Calculate "RawChange"
rawChangeTKA = TKApainpost - TKApainpre;
ratioTKA = (TKApainpre - TKApainpost) ./ (TKApainpost + TKApainpre);

% Calculate "RawChange"
rawChangeWO = WOpainpost - WOpainpre;
ratioWO = (WOpainpre - WOpainpost) ./ (WOpainpost + WOpainpre);

% GM2 = numericData(:, strcmp(numericTitles, 'GM2'));
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);

% Fit linear models
mdl = fitglm(rawChangeTKA, TKApainpre);
rawChangeTKAadj = mdl.Residuals.Raw;

mdl = fitglm(rawChangeWO, WOpainpre);
rawChangeWOadj = mdl.Residuals.Raw;

%% PREDICTIONS

ALL = [rawChangeTKA, rawChangeTKAadj, ratioTKA, rawChangeWO, rawChangeWOadj, ratioWO, TKApainpre, WOpainpre, ROIs, genotype];
ALL = normvalues(ALL);

% Define the pattern
pattern = 'SC|CC';

% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub = [numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'painpre')))), numericTitles(indices), numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];

targetarray = ALL(:, 1:6);
k = 6;

if k == 1
    varname = 'RawChange TKA';
    input = ALL(:, 7);
    predictorsCombined = ALL(:, [7, 9:end]);
elseif k == 2
    varname = 'RawChange TKA adj';
    input = ALL(:, 7);
    predictorsCombined = ALL(:, [7, 9:end]);
elseif k == 3
    varname = 'Normalised Improvement TKA';
    input = ALL(:, 7);
    predictorsCombined = ALL(:, [7, 9:end]);
elseif k == 4
    varname = 'RawChange WO';
    input = ALL(:, 8);
    predictorsCombined = ALL(:, 8:end);
elseif k == 5
    varname = 'RawChange WO adj';
    input = ALL(:, 8);
    predictorsCombined = ALL(:, 8:end);
elseif k == 6
    varname = 'Normalised Improvement WO';
    input = ALL(:, 8);
    predictorsCombined = ALL(:, 8:end);
end

target = targetarray(:, k);
options = statset('UseParallel', true);

optimizeFeatures = false;

if optimizeFeatures
    % Optimize for the optimal number of features
    selectedFeaturesidx = sequentialfs(@critfun, predictorsCombined, target, 'cv', 'none', 'options', options);
else
    % Choose a fixed number of features
    numSelectedFeatures = 20; % Choose the desired number of features
    selectedFeaturesidx = sequentialfs(@critfun, predictorsCombined, target, 'cv', 'none', 'options', options, 'NFeatures', numSelectedFeatures);
end

selectedFeatures = predictorsCombined(:, selectedFeaturesidx);

selectedFeaturesSTORE = zeros(size(predictorsCombined, 2), 1);
STORE = [];

% Initialize result containers
predictions1 = zeros(1, numel(target));
predictions2 = zeros(1, numel(target));

% Loop for leave-one-out cross-validation
for i = 1:numel(target)
    % Create training and testing sets
    targetTrain = target;
    targetTrain(i) = [];
    targetTest = target(i);
    
    inputTrain = input;
    inputTrain(i) = [];
    inputTest = input(i);

    predictorsCombinedTrain = selectedFeatures;
    predictorsCombinedTrain(i, :) = [];
    
    predictorsCombinedTest = selectedFeatures(i, :);

    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain, targetTrain);
    predictions1(i) = predict(mdl1, inputTest);

    % Fit linear model with selected predictors for combined vs target
    mdlCombined = fitlm(predictorsCombinedTrain, targetTrain);
    predictions2(i) = predict(mdlCombined, predictorsCombinedTest);
    
    % Store the selected features for the current fold
    selectedFeaturesSTORE(selectedFeaturesidx) = 1;
    STORE = [STORE, selectedFeaturesSTORE];
    % Reset selectedFeaturesSTORE for the next fold
    selectedFeaturesSTORE = zeros(size(predictorsCombined, 2), 1);

    fprintf('Progress: %.2f%%\n', 100 * i / numel(target));
end

% Plot correlations and regression lines
figure(1)
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(target, predictions1', 30, 'b');
title({"Model: TkaPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
ylabel('Predicted');

ylabel('Predicted');
xlabel('True');
hold off;

subplot(1, 2, 2);
[rho2, p2] = PlotSimpleCorrelationWithRegression(target, predictions2', 30, 'b');
title({"Model: [TkaPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
hold off;

% Sum the frequencies across folds
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
