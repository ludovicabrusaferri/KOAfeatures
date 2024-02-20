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
sex = numericData(:, strcmp(numericTitles, 'Sex (1=F)'));
TKApainpre = numericData(:, strcmp(numericTitles, 'TKA pain pre'));
TKApainpost = numericData(:, strcmp(numericTitles, 'TKA pain post'));
WOpainpre = numericData(:, strcmp(numericTitles, 'Pre-TKA,WOMAC (pain)'));
WOphyspre = numericData(:, strcmp(numericTitles, 'Pre-TKA,WOMAC (physical function)'));
WOstiffpre = numericData(:, strcmp(numericTitles, 'Pre-TKA,WOMAC (stiffness)'));
WOpainpost = numericData(:, strcmp(numericTitles, '1yr POST-TKA, Womac (pain)'));
WOphyspost = numericData(:, strcmp(numericTitles, '1yr POST-TKA,Womac (phys func)'));
WOstiffpost = numericData(:, strcmp(numericTitles, '1yr POST-TKA,Womac (stiffness)'));
WOtotpre = WOpainpre +WOphyspre + WOstiffpre;
WOtotpost = WOpainpost +WOphyspost + WOstiffpost;

Promispre = numericData(:, strcmp(numericTitles, 'PRE-TKA, Promis (pain intensity)'));


% Small epsilon value for numerical stability
eps = 0.00000000001;

% Calculate "RawChange" and corresponding ratio for TKA
rawChangeTKA = TKApainpost - TKApainpre;
% Calculate "RawChange" and corresponding ratio for WOMAC
rawChangeWO = WOpainpost - WOpainpre;

ratioTKA = ratioimpr(TKApainpre,TKApainpost,eps);
ratioWOpain = ratioimpr(WOpainpre,WOpainpost,eps);
ratioWOstiff = ratioimpr(WOstiffpre,WOstiffpost,eps);
ratioWOphys = ratioimpr(WOphyspre,WOphyspost,eps);
ratioWOtot = ratioimpr(WOtotpre,WOtotpost,eps);

% Extract SC and CC variables for further analysis
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);



%% PREDICTIONS

% Specify the variable name you want to use
optionname = 'pain';
% Construct the variable names using the specified option
variableName = ['ratioWO' optionname];
preVariableName = ['WO' optionname 'pre'];
% Use the constructed variable names in your code
ALL = [eval(variableName), eval(preVariableName), genotype, sex, ROIs];
ALL = normvalues(ALL);
varname = ['Norm. Impr WO ' optionname];
modelname = [preVariableName, {'geno'}, {'sex'}];

pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub = numericTitles(indices);
titlen = [modelname,numericTitles_sub]';

modelname = [modelname, {'ROIs'}];

% Convert the cell array to a single string
modelname = strjoin(modelname, ' ');


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

STORE = [];


% Leave-one-out cross-validation loop
for fold = (1:numFolds)
    % Reset selectedFeaturesSTORE for the next fold
    selectedFeaturesSTORE = zeros(size(input, 2), 1);

    % Calculate the indices for training and testing sets
    testIndices = round((fold - 1) * numel(target) / numFolds + 1 : fold * numel(target) / numFolds);
    trainIndices = setdiff(1:numel(target), testIndices);

    % Create training and testing sets
    targetTrain = target(trainIndices);
    targetTest = target(testIndices);
    
    inputTrain = input(trainIndices, :);
    inputTest = input(testIndices, :);
   
    % Feature selection using RFE with SVM
    mdl = fitrsvm(inputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
    featureWeights = abs(mdl.Beta);
    [~, sortedIndices] = sort(featureWeights, 'descend');
    selectedFeaturesidx = sortedIndices(1:numSelectedFeatures);
    selectedFeaturesSTORE(selectedFeaturesidx) = 1;
    STORE = [STORE, selectedFeaturesSTORE];

    % Train the final model using selected features
    selectedInputTrain = inputTrain(:, selectedFeaturesidx);
    selectedInputTest = inputTest(:, selectedFeaturesidx);
    
    combinedModel = fitrsvm(selectedInputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
    % Make predictions on the test data
    predictedTarget(testIndices) = predict(combinedModel, selectedInputTest);

    
    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain(:, 1), targetTrain);
    predictions1(testIndices) = predict(mdl1, inputTest(1));

    % Evaluate the performance (you can replace this with your own evaluation metrics)
    fprintf('Progress: %.2f%%\n', 100 * testIndices/ numel(target));
end

%%
figure(2)
subplot(1,2,1)
[rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictions1', 30, 'b');
title({sprintf("Model: %s",preVariableName), sprintf("vs %s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1.3])
ylim([0,1.3])
hold off
subplot(1,2,2)
[rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictedTarget', 30, 'b');
title({sprintf("Model: %s", modelname), sprintf("vs %s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
xlim([0,1.3])
ylim([0,1.3])
hold off


%% FREQ

% Sum the frequencies across folds for selected features
freq = sum(STORE, 2);



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
figure(3)
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
%%

variableName = [{'ratioWOpain'},{'ratioWOstiff'},{'ratioWOphys'},{'ratioWOtot'}];
figure(11)

subplot(1,3,1)
[rho2, p2] = PlotSimpleCorrelationWithRegression(eval(variableName{1}), eval(variableName{2}), 30, 'b');
title({sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel(variableName(2));
xlabel(variableName(1));

subplot(1,3,2)
[rho2, p2] = PlotSimpleCorrelationWithRegression(eval(variableName{1}), eval(variableName{3}), 30, 'b');
title({sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel(variableName(3));
xlabel(variableName(1));

subplot(1,3,3)
[rho2, p2] = PlotSimpleCorrelationWithRegression(eval(variableName{1}), eval(variableName{4}), 30, 'b');
title({sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel(variableName(4));
xlabel(variableName(1));




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

function ratio = ratioimpr(pre, post, eps)
    ratio = (pre - post) ./ (post + pre + eps);
end

