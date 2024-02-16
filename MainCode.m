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

% Set options for feature selection
options = statset('UseParallel', true);
alwaysIncludeFirst = false;
selectOutside = false;

% Initialize result containers
predictions1 = zeros(1, numel(target));
predictions2 = zeros(1, numel(target));
selectedFeaturesSTORE = zeros(size(input, 2), 1);
STORE = [];

if selectOutside
    method = "SequentialsFixed"; % Specify the method
    linearmodel = true;
    numSelectedFeatures = 30; % Choose the desired number of features
    selectedFeaturesidx = selectedFeatures(input, target, method, numSelectedFeatures, options, alwaysIncludeFirst);   
end

% Set the number of folds
numFolds = 41;

for fold = randperm(numFolds)
    % Calculate the indices for training and testing sets
    testIndices = round((fold - 1) * numel(target) / numFolds + 1 : fold * numel(target) / numFolds);
    trainIndices = setdiff(1:numel(target), testIndices);

    % Create training and testing sets
    targetTrain = target(trainIndices);
    targetTest = target(testIndices);
    
    inputTrain = input(trainIndices, :);
    inputTest = input(testIndices, :);

    if selectOutside
        inputTrainSelected = inputTrain(:, selectedFeaturesidx);
        inputTestSelected = inputTest(:, selectedFeaturesidx);
    
    else
        method = "SVM"; % Specify the method
        linearmodel = false;
        numSelectedFeatures = 15; % Choose the desired number of features
        selectedFeaturesidx = selectedFeatures(inputTrain, targetTrain, method, numSelectedFeatures, options, alwaysIncludeFirst);   
        inputTrainSelected = inputTrain(:, selectedFeaturesidx);
        inputTestSelected = inputTest(:, selectedFeaturesidx);
    end

    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain(:, 1), targetTrain);
    predictions1(testIndices) = predict(mdl1, inputTest(1));
    
    % Fit linear model or SVM with selected predictors for combined vs target
    if linearmodel
        mdlCombined = fitlm(inputTrainSelected, targetTrain);
        %mdlCombined = fitrlinear(inputTrainSelected, targetTrain, 'Regularization', 'ridge', 'Lambda', 'auto');

    else
        %mdlCombined = fitrsvm(inputTrainSelected, targetTrain, 'Standardize', true,  'KernelFunction', 'linear');
        mdlCombined = fitrsvm(inputTrainSelected, targetTrain, 'KernelFunction','linear', 'Standardize',true);
    end
    
    predictions2(testIndices) = predict(mdlCombined, inputTestSelected);
    
    % Store the selected features for the current fold
    selectedFeaturesSTORE(selectedFeaturesidx) = 1;
    STORE = [STORE, selectedFeaturesSTORE];
    % Reset selectedFeaturesSTORE for the next fold
    selectedFeaturesSTORE = zeros(size(input, 2), 1);
    fprintf('Progress: %.2f%%\n', 100 * testIndices / numel(target));
end

%% PLOT RESULTS


figure(11)
[rho2, p2] = PlotSimpleCorrelationWithRegression(targetTrain, predict(mdlCombined, inputTrainSelected), 30, 'b');
titleText = {"Training: [PainPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)};
%title('hi')
title({"Training: [PainPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0.0,1.5])
hold off


%%
%% PLOT RESULTS

close all
% Plot correlations and regression lines for input vs target and combined vs target
figure(1)
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(target,  predictions1', 30, 'b');
title({"Model: PainPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0.3,1.5])
hold off

subplot(1, 2, 2);
[rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictions2', 30, 'b');
title({"Model: [PainPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1])
ylim([0.3,1.5])
hold off
%%
% Sum the frequencies across folds for selected features
freq = sum(STORE, 2);

pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub = numericTitles(indices);
titlen = ['WOMAC Pain Pre' ,'genotype',numericTitles_sub]';

% Convert freq to a cell array
freqCell = num2cell(freq);

% Concatenate freqCell and title
combinedData = [freqCell, titlen];

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

function out = standardizeValues(input)
    [rows, cols] = size(input);
    out = zeros(rows, cols);

    for col = 1:cols
        % Standardize each column separately using zscore, ignoring NaN values
        colData = input(:, col);
        nonNaNIndices = ~isnan(colData);
        
        if any(nonNaNIndices)
            % Only standardize if there are non-NaN values in the column
            out(nonNaNIndices, col) = zscore(colData(nonNaNIndices));
        else
            % If all values in the column are NaN, leave them as they are
            out(:, col) = colData;
        end
    end
end




function selectedFeaturesidx = selectedFeatures(input, target, method, numSelectedFeatures, options, alwaysIncludeFirst)
% Feature selection based on the chosen method
    if strcmp(method, 'SequentialsVariable')
        % Optimize for the optimal number of features
        selectedFeaturesidx = sequentialfs(@critfun, input, target, 'cv', 'none', 'options', options);
    elseif strcmp(method, 'SequentialsFixed')
        % Choose a fixed number of features
        selectedFeaturesidx = sequentialfs(@critfun, input, target, 'cv', 'none', 'options', options, 'NFeatures', numSelectedFeatures);
    elseif strcmp(method, 'Lasso')
        % L1 regularization (LASSO) for feature selection
        [B, FitInfo] = lasso(iinput, target, 'CV', 20); % You can adjust 'CV' as needed
        idxLambdaMinMSE = FitInfo.IndexMinMSE;
        selectedFeaturesidx = B(:, idxLambdaMinMSE) ~= 0;
    elseif strcmp(method, 'RandomForest')
        % Train Random Forest with OOBPermuteVarDeltaError
        numTrees = 30; % You can adjust the number of trees
        baggedTree = TreeBagger(numTrees, input, target, 'Method', 'regression', 'OOBPredictorImportance', 'on');
        importance = baggedTree.OOBPermutedVarDeltaError;
        [~, sortedIdx] = sort(importance, 'descend');
        selectedFeaturesidx = sortedIdx(1:numSelectedFeatures);
    elseif strcmp(method, "SVM")
        % SVM-based feature selection for regression
        svmModel = fitrsvm(input, target, 'KernelFunction','linear','Standardize',true);
        featureWeights = abs(svmModel.Beta);
        [~, sortedIndices] = sort(featureWeights, 'descend');
        selectedFeaturesidx = sortedIndices(1:numSelectedFeatures);

        % Check if features 1 and 2 are already selected
        includeFeature1 = ~any(selectedFeaturesidx == 1);
        includeFeature2 = ~any(selectedFeaturesidx == 2);
    
        % Update selectedFeaturesidx based on the checks
        if alwaysIncludeFirst && includeFeature1
            selectedFeaturesidx = [1; selectedFeaturesidx];
        end
        if alwaysIncludeFirst && includeFeature2
            selectedFeaturesidx = [2; selectedFeaturesidx];
        end
    
        % Include the top numSelectedFeatures - 2 features without the first two
        selectedFeaturesidx = [selectedFeaturesidx; sortedIndices(1:numSelectedFeatures - length(selectedFeaturesidx))];
    end
end