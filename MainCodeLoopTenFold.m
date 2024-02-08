% Clear workspace and command window
clear all;
clc;

% Set the data file path
%dataFilePath = '/Users/luto/Dropbox/KOAfeatures/ForLudoo.xlsx';
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


mdl=fitglm(rawChange,TKApainpre);
rawChangeAdj=mdl.Residuals.Raw;
%% %% ====== PREDICTIONSß ===========
ALL = [rawChangeAdj, ratio, TKApainpre, ROIs, genotype];

% Set target and predictors for regression
targetarray = ALL(:, 1:2);
k = 1;
if k == 1
    varname = 'RawChange residuals';
elseif k == 2
    varname = 'Normalised Improvement';
end

target = targetarray(:, k);
input = ALL(:, 3);
predictorsCombined =  ALL(:, 3:end);

pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub = numericTitles(indices);
numericTitles_sub = [numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'painpre')))), numericTitles_sub, numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];

% Initialize result containers
predictions1 = zeros(1, numel(target));
predictions2 = zeros(1, numel(target));

% Loop for 10-fold cross-validation
numFolds = 10;
foldSize = numel(target) / numFolds;

for fold = 1:numFolds
    % Determine indices for the current fold
    foldStart = round((fold - 1) * foldSize) + 1;
    foldEnd = round(fold * foldSize);

    % Create training and testing sets
    targetTrain = target;
    targetTrain(foldStart:foldEnd) = [];
    targetTest = target(foldStart:foldEnd);

    inputTrain = input;
    inputTrain(foldStart:foldEnd) = [];
    inputTest = input(foldStart:foldEnd);

    predictorsCombinedTrain = predictorsCombined;
    predictorsCombinedTrain(foldStart:foldEnd, :) = [];

    % Lasso regularization for feature selection
    numSelectedFeatures = 30; % Choose the desired number of features
    options = statset('UseParallel', true);
    selectedFeaturesidx = sequentialfs(@critfun, predictorsCombinedTrain, targetTrain, 'cv', 'none', 'options', options, 'Nfeatures', numSelectedFeatures);
    
    % Use the selected features for modeling
    selectedFeatures = predictorsCombinedTrain(:, selectedFeaturesidx);
    selectednumericTitles_sub = numericTitles_sub(:, selectedFeaturesidx);

    % Initialize result containers for the current fold
    foldPredictions1 = zeros(1, numel(targetTest));
    foldPredictions2 = zeros(1, numel(targetTest));

    % Feature indices
    idx = [];
    for j = 1:size(selectedFeatures, 2)
        for k = 1:size(predictorsCombinedTrain, 2)
            if isequal(predictorsCombinedTrain(:, k), selectedFeatures(:, j))
                idx = [idx, k];
            end
        end
    end

    predictorsCombinedTestALL = predictorsCombined(foldStart:foldEnd, idx);
    predictorsCombinedTest = predictorsCombinedTestALL;

    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain, targetTrain);
    foldPredictions1 = predict(mdl1, inputTest);

    % Regularization parameter
    %lambda = 0.1;  % You can adjust the value of lambda
    
    % Add regularization term to the predictors and target for training
    %regularizedPredictors = [selectedFeatures; sqrt(lambda) * eye(size(selectedFeatures, 2))];
    %regularizedTarget = [targetTrain; zeros(size(selectedFeatures, 2), 1)];
    
    % Fit linear model with regularization
    mdlCombined = fitlm(selectedFeatures, targetTrain);
    foldPredictions2 = predict(mdlCombined, predictorsCombinedTest);

    % Assign fold predictions to the overall predictions
    predictions1(foldStart:foldEnd) = foldPredictions1;
    predictions2(foldStart:foldEnd) = foldPredictions2;

    fprintf('DONE with Fold %d\n', fold);
end

% Plot correlations and regression lines for the entire dataset
figure;
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(target, predictions1', 30, 'b');
title({"Model: TkaPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
ylabel('Predicted');
xlabel('True');
hold off;

subplot(1, 2, 2);
[rho2, p2] = PlotSimpleCorrelationWithRegression(target, predictions2', 30, 'b');
title({"Model: [TkaPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
hold off;




%% plot training

% Plot correlations and regression lines
figure(2)
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

function crit = critfun(x, y)
    mdl = fitlm(x, y);
    crit = mdl.RMSE;
end


