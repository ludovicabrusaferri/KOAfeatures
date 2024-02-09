% Clear workspace and command window
clear all;
clc;
rng(43);
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
WOstiffpre = numericData(:, strcmp(numericTitles, 'Pre-TKA,WOMAC (stiffness)'));
WOpainpost = numericData(:, strcmp(numericTitles, '1yr POST-TKA, Womac (pain)'));
PRpainpre = numericData(:, strcmp(numericTitles, 'PRE-TKA, Promis (pain intensity)'));
PRpainpost = numericData(:, strcmp(numericTitles, '1yr POST-TKA, Promis (pain intensity)'));

eps = 0.000000000000001;
% Calculate "RawChange"
rawChangeTKA = TKApainpost - TKApainpre;
ratioTKA = (TKApainpre - TKApainpost) ./ (TKApainpost + TKApainpre + eps);

% Calculate "RawChange"
rawChangeWO = WOpainpost - WOpainpre;
ratioWO = (WOpainpre - WOpainpost) ./ (WOpainpost + WOpainpre + eps);

% Calculate "RawChange"
rawChangePR = PRpainpost - PRpainpre;
ratioPR = (PRpainpre - PRpainpost) ./ (PRpainpost + PRpainpre + eps);

% GM2 = numericData(:, strcmp(numericTitles, 'GM2'));
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);

mdl = fitglm(rawChangeTKA, TKApainpre);
rawChangeTKAadj = mdl.Residuals.Raw;
mdl = fitglm(rawChangeWO, WOpainpre);
rawChangeWOadj = mdl.Residuals.Raw;
mdl = fitglm(rawChangePR, PRpainpre);
rawChangePRadj = mdl.Residuals.Raw;

%% PREDICTIONS

ALL = [ratioWO, WOpainpre, ROIs, genotype];

% Define the pattern
pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub = numericTitles(indices);

varname = 'Normalised Improvement WO';
ALL(isnan(ALL(:, 1)), :) = [];
input = ALL(:, 2);
predictorsCombined = ALL(:, 2:end);
%numericTitles_sub=['Pre-TKA,WOMAC (pain)',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
target = ALL(:, 1);


plot_training = 0;

% Initialize result containers
predictions1 = zeros(1, numel(target));
predictions2_lasso = zeros(1, numel(target));

P = {};
numFolds = round(1 * numel(target)); % Number of folds for leave-one-out cross-validation
% Initialize a matrix to store selected features frequencies
% Initialize a cell array to store selected features for each fold
% Initialize a matrix to store selected features frequencies
selectedFeaturesSTORE = zeros(size(predictorsCombined, 2), 1);
STORE = [];
%%
% Loop for leave-one-out cross-validation
for fold = 1:numFolds
    % Indices for the current fold
    testIndices = false(1, numFolds);
    testIndices(fold) = true;

    % Create training and testing sets
    targetTrain = target(~testIndices);
    inputTrain = input(~testIndices);
    predictorsCombinedTrain = predictorsCombined(~testIndices, :);

    targetTest = target(testIndices);
    inputTest = input(testIndices);
    predictorsCombinedTestALL = predictorsCombined(testIndices, :);

    % Feature engineering
    %polyDegree = 2;
    %polyFeatures = inputTrain.^polyDegree;
    inputTrain = [inputTrain];%, polyFeatures];

    %polyFeaturesTest = inputTest.^polyDegree;
    inputTest = [inputTest];%, polyFeaturesTest];

    % Fit linear model with input vs target
    mdl1 = fitlm(inputTrain, targetTrain);

    % Fit LASSO to select important features
    [B_lasso, FitInfo_lasso] = lasso(predictorsCombinedTrain, targetTrain, 'CV', 20);

    % Identify the optimal lambda from cross-validation
    lambda_optimal = FitInfo_lasso.LambdaMinMSE;

    % Extract the nonzero coefficients from the optimal model
    selectedFeaturesIdx = find(B_lasso(:, FitInfo_lasso.IndexMinMSE) ~= 0);
    selectedFeatures = predictorsCombinedTrain(:, selectedFeaturesIdx);

    % Apply the selected features to the testing set
    predictorsCombinedTest_lasso = predictorsCombinedTestALL(:, selectedFeaturesIdx);

    % Fit linear model with selected LASSO predictors for combined features vs target
    mdlCombined_lasso = fitlm(selectedFeatures, targetTrain);

    % Predictions using the linear model
    predictions1(testIndices) = predict(mdl1, inputTest);

    % Predictions using the LASSO-selected features
    predictions2_lasso(testIndices) = predict(mdlCombined_lasso, predictorsCombinedTest_lasso);

    fprintf('DONE.. Fold %d\n', fold);

    % Store the selected features for the current fold
    selectedFeaturesSTORE(selectedFeaturesIdx) = 1;
    STORE = [STORE, selectedFeaturesSTORE];
    fprintf('DONE.. Fold %d\n', fold);

    % Reset selectedFeaturesSTORE for the next fold
    selectedFeaturesSTORE = zeros(size(predictorsCombined, 2), 1);
end
%STORE=STORE';

%%
% Plot correlations and regression lines
figure(1)
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(target, predictions1', 30, 'b');
title({"Model: TkaPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});

ylabel('Predicted');
xlabel('True');
hold off;
ylim([-3, 3])
xlim([-1,1])
subplot(1, 2, 2);
[rho2_lasso, p2_lasso] = PlotSimpleCorrelationWithRegression(target, predictions2_lasso', 30, 'b');
title({"Model: [TkaPre, ROIs, geno] (LASSO) vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2_lasso, p2_lasso)});
ylabel('Predicted');
xlabel('True');
hold off;
ylim([-3, 3])
xlim([-1,1])


%%
% Sum the frequencies across folds
freq = sum(STORE, 2);

% Plot histogram of selected features frequencies
figure(2);
bar(freq);
xlabel('Feature Index');
ylabel('Frequency across Folds');
title('Selected Features Frequencies over Folds');
set(gcf, 'Color', 'w');
set(gca,'FontSize',25)
%%

%% plot training
plot_training=0;
if (plot_training)
    % Plot correlations and regression lines
    figure(4)
    subplot(1, 2, 1);
    [rho1, p1] = PlotSimpleCorrelationWithRegression(targetTrain, mdl1.Fitted, 30, 'b');
    title({"Training: TkaPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
    ylabel('Predicted');
    
    ylabel('Predicted');
    xlabel('True');
    hold off;
    
    subplot(1, 2, 2);
    [rho2, p2] = PlotSimpleCorrelationWithRegression(targetTrain, mdlCombined.Fitted, 30, 'b');
    title({"Training: [TkaPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
    ylabel('Predicted');
    xlabel('True');
    hold off;
    
    %% ====== CLUSTERING =========== (let's see what to do here...)
    
    % Combine variables for clustering
    dataForClustering = [TKApainpre, rawChangeTKA];
    
    % Normalize the data between min and max
    dataForClustering = normalize(dataForClustering, 'range');
    
    % Specify the number of clusters (k=4)
    numClusters = 4;
    
    % Set the random seed for reproducibility
    rng(42);
    
    % Run k-means clustering
    [idx, centers] = kmeans(dataForClustering, numClusters);
    
    % Plot the results
    figure;
    for i = 1:numClusters
        clusterIndices = idx == i;
        scatter(dataForClustering(clusterIndices, 1), dataForClustering(clusterIndices, 2), 40, 'filled', 'DisplayName', ['Cluster ' num2str(i)]);
        hold on;
    end
    
    scatter(centers(:, 1), centers(:, 2), 200, 'X', 'LineWidth', 2, 'DisplayName', 'Cluster Centers');
    
    % Add legend and labels
    legend('show');
    title('K-Means Clustering (Normalized)');
    xlabel('Normalized TKApainpre');
    ylabel('Normalized RawChange');
    set(gca, 'FontSize', 25);
    hold off;
    set(gcf, 'Color', 'w');
    %%

    else
    fprintf('hii')
end

function crit = critfun(x, y)
    mdl = fitlm(x, y);
    crit = mdl.RMSE;
end