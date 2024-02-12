% Clear workspace and command window
clear all;
clc;
rng(46);
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

eps = 0.00000000001;
% Calculate "RawChange"
rawChangeTKA = TKApainpost - TKApainpre;
ratioTKA = (TKApainpre - TKApainpost) ./ (TKApainpost + TKApainpre + eps);

% Calculate "RawChange"
rawChangeWO = WOpainpost - WOpainpre;
ratioWO = (WOpainpre - WOpainpost) ./ (WOpainpost + WOpainpre + eps);

% GM2 = numericData(:, strcmp(numericTitles, 'GM2'));
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);

% Fit linear models
mdl = fitglm(rawChangeTKA, TKApainpre); rawChangeTKAadj = mdl.Residuals.Raw;
mdl = fitglm(rawChangeWO, WOpainpre); rawChangeWOadj = mdl.Residuals.Raw;

%% PREDICTIONS
% ... (previous code)

% PREDICTIONS
ALL = [ratioWO, WOpainpre, genotype, ROIs];
ALL = normvalues(ALL);

varname = 'Normalised Improvement WO';
target = ALL(:, 1);
ALL(isnan(target), :) = [];
input = ALL(:, 2);
predictorsCombined = ALL(:, 2:end);

predictorsCombined = [predictorsCombined];

% ... (rest of the code)

%%
options = statset('UseParallel', true);

optimizeFeatures = false;

if optimizeFeatures
    % Optimize for the optimal number of features
    selectedFeaturesidx = sequentialfs(@critfun, predictorsCombined, target, 'cv', 'none', 'options', options);
else
    % Choose a fixed number of features
    numSelectedFeatures = 30; % Choose the desired number of features
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
figure(3)
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(target, predictions1', 30, 'b');
title({"Model: PainPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
ylabel('Predicted');

ylabel('Predicted');
xlabel('True');
hold off;

subplot(1, 2, 2);
[rho2, p2] = PlotSimpleCorrelationWithRegression(target, predictions2', 30, 'b');
title({"Model: [PainPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
hold off;

% Sum the frequencies across folds
freq = sum(STORE, 2);

%%
% Sum the frequencies across folds for selected features
freq = sum(STORE, 2);

pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub = numericTitles(indices);
title = ['WOMAC Pain Pre','genotype' ,numericTitles_sub]';

% Convert freq to a cell array
freqCell = num2cell(freq);

% Concatenate freqCell and title
combinedData = [freqCell, title];

% Sort based on the first column (freq) in descending order
sortedData = sortrows(combinedData, -1);

% Extract the sorted freq and title
sortedFreq = cell2mat(sortedData(:, 1));
sortedTitle = sortedData(:, 2);
% Plot the histogram using the histogram function
% Plot the histogram
% Convert the cell array of titles to a cell array of strings
sortedTitleStrings = cellfun(@str2mat, sortedTitle, 'UniformOutput', false);

% Plot the histogram using the histogram function
figure(2)
bar(sortedFreq);

% Set the x-axis labels to be the sorted titles
xticks(1:length(sortedFreq));
xticklabels(sortedTitleStrings);

% Rotate x-axis labels for better visibility
xtickangle(45);

% Set axis labels and title
xlabel('Titles');
ylabel('Frequency');
%title('Histogram of Sorted Data');
set(gcf, 'Color', 'w');
set(gca,'FontSize',15)

% Display the plot
grid on;


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
