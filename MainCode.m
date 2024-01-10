% Clear workspace and command window
clear all;
clc;

% Set the data file path
dataFilePath = '/Users/e410377/Desktop/KOAfeatures/ForLudoo.xlsx';

% Import data from Excel file
data = importdata(dataFilePath);

% Extract numeric data and corresponding titles
numericData = data.data;
numericTitles = data.textdata(1, 2:end);

% Extract relevant variables
age = numericData(:, strcmp(numericTitles, 'Age (pre)'));
genotype = numericData(:, strcmp(numericTitles, 'Genotype (1=GG)'));
TKApainpre = numericData(:, strcmp(numericTitles, 'TKA painpre'));
TKApainpost = numericData(:, strcmp(numericTitles, 'TKA painpost'));

% Calculate "RawChange"
rawChange = TKApainpost - TKApainpre;
% Load and process additional data
ratio = numericData(:, strcmp(numericTitles, 'RATIO'));
GM2 = numericData(:, strcmp(numericTitles, 'GM2'));
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);



%% %% ====== PREDICTIONSß ===========

% Set target and predictors for regression
targetarray = [rawChange,ratio];
k = 1;
if k==1
    varname = 'RawChange';
elseif k==2
    varname = 'Normalised Improvement';
end

target=targetarray(:,k);
input = TKApainpre;
predictorsCombined = [input, ROIs, genotype];

% Lasso regularization for feature selection
[BCombined, FitInfoCombined] = lasso(predictorsCombined, target); % can this be oustide the loop?

% Identify non-zero coefficients (selected features)
selectedFeatures = predictorsCombined(:, BCombined(:, 1) ~= 0);

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
end


% Plot correlations and regression lines
figure;
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


%% ====== CLUSTERING =========== (let's see what to do here...)

% Combine variables for clustering
dataForClustering = [TKApainpre, rawChange];

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