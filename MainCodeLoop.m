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
var=rawChangeAdj; rawChangeAdj2=(var - min(var))./(max(var) - min(var));
var=TKApainpre;TKApainpre2=(var - min(var))./(max(var) - min(var));
ALL=[rawChangeAdj2,ratio,TKApainpre2,ROIs, genotype];

%rowsToKeep = ratio~=1;

%ALL(~rowsToKeep, :) = [];
% Set target and predictors for regression
targetarray = ALL(:,1:2);
k =2;
if k==1
    varname = 'RawChange residuals';
elseif k==2
    varname = 'Normalised Improvement';
end

target=targetarray(:,k);
input = ALL(:,3);
predictorsCombined =  ALL(:,3:end);

pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub=numericTitles(indices);
numericTitles_sub=[numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'painpre')))),numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];


% Initialize result containers
predictions1 = zeros(1, numel(target));
predictions2 = zeros(1, numel(target));
P={};
% Loop for leave-one-out cross-validation
for i = 1:numel(target)
    % Create training and testing sets
    targetTrain = target;
    targetTrain(i) = [];
    targetTest = target(i);
    
    inputTrain = input;
    inputTrain(i) = [];
    inputTest = input(i);
    
    predictorsCombinedTrain = predictorsCombined;

    predictorsCombinedTrain(i, :) = [];

    % Lasso regularization for feature selection
    %lambda = logspace(-0.2, 2, 100);
    %[BCombined, FitInfoCombined] = lasso(predictorsCombinedTrain, targetTrain); % can this be oustide the loop?
    % Example for Elastic Net
    
    % Elastic Net regularization for feature selection
    %alpha = 0.0001; % You can adjust the 'Alpha' parameter based on your needs
    %[BCombined, FitInfoCombined] = lasso(predictorsCombinedTrain, targetTrain);

    % Identify non-zero coefficients (selected features)
    %selectedFeatures = predictorsCombinedTrain(:, BCombined(:, 1) ~= 0);
    % Perform feature selection using recursive feature elimination
    numSelectedFeatures = 10; % Choose the desired number of features
    
    options = statset('UseParallel',true);
    selectedFeaturesidx = sequentialfs(@critfun, predictorsCombinedTrain, targetTrain, 'cv', 'none', 'options', options, 'Nfeatures', numSelectedFeatures);
    
    % Use the selected features for modeling
    selectedFeatures = predictorsCombinedTrain(:, selectedFeaturesidx);
    selectednumericTitles_sub=numericTitles_sub(:, selectedFeaturesidx);

    idx= [];
    for j = 1:size(selectedFeatures,2)
        for k = 1:size(predictorsCombinedTrain,2)
            if predictorsCombinedTrain(:, k)==selectedFeatures(:, j)
                idx=[idx,k];
            end
        end
    end
    
    predictorsCombinedTestALL = predictorsCombined(:,idx);
    predictorsCombinedTest = predictorsCombinedTestALL(i, :);


    % Fit linear model with selected predictors for input vs target
    mdl1 = fitlm(inputTrain, targetTrain);
    predictions1(i) = predict(mdl1, inputTest);

    % Fit linear model with selected predictors for combined vs target
    mdlCombined = fitlm(selectedFeatures, targetTrain);
    predictions2(i) = predict(mdlCombined, predictorsCombinedTest);
    fprintf('DONE..%d\n',i);
    P=[P;selectednumericTitles_sub];
end


% Plot correlations and regression lines
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


%%
% Assuming numericTitles_sub is a 1x71 cell array of strings and P_flattened is a cell array of strings

% Flatten P into a column cell array of strings
P_flattened = P(:);

% Initialize a container for counting occurrences
occurrences = zeros(size(numericTitles_sub));

% Loop over each entry in numericTitles_sub
for i = 1:numel(numericTitles_sub)
    % Count occurrences of the current entry in P_flattened
    occurrences(i) = sum(strcmp(numericTitles_sub{i}, P_flattened));
end

% Sort occurrences and get the corresponding indices
[sorted_occurrences, sorted_indices] = sort(occurrences, 'descend');

% Plot the frequency histogram in descending order
figure;
bar(1:numel(numericTitles_sub), sorted_occurrences, 'BarWidth', 0.8);
xticks(1:numel(numericTitles_sub));
xticklabels(numericTitles_sub(sorted_indices));
xlabel('Numeric Titles Sub');
ylabel('Frequency');
title('Frequency Plot of Numeric Titles Sub in P (Descending Order)');


%%
% Plot the histogram
figure;
bar(unique_values, frequency, 'BarWidth', 0.8);
xlabel('Unique Values in P');
ylabel('Frequency');
title('Frequency Histogram of Values in P with respect to numericTitles_sub');
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

