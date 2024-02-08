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
PRpainpre = numericData(:, strcmp(numericTitles, 'PRE-TKA, Promis (pain intensity)'));
PRpainpost = numericData(:, strcmp(numericTitles, '1yr POST-TKA, Promis (pain intensity)'));

eps = 0.000000000000001;
% Calculate "RawChange"
rawChangeTKA = TKApainpost - TKApainpre;
ratioTKA = (TKApainpre-TKApainpost)./(TKApainpost + TKApainpre+eps);

% Calculate "RawChange"
rawChangeWO = WOpainpost - WOpainpre;
ratioWO = (WOpainpre-WOpainpost)./(WOpainpost + WOpainpre+eps);

% Calculate "RawChange"
rawChangePR = PRpainpost - PRpainpre;
ratioPR = (PRpainpre-PRpainpost)./(PRpainpost + PRpainpre+eps);


%GM2 = numericData(:, strcmp(numericTitles, 'GM2'));
SC = numericData(:, strncmp(numericTitles, 'SC', 2));
CC = numericData(:, strncmp(numericTitles, 'CC', 2));
ROIs = cat(2, SC, CC);


mdl=fitglm(rawChangeTKA,TKApainpre);
rawChangeTKAadj=mdl.Residuals.Raw;
mdl=fitglm(rawChangeWO,WOpainpre);
rawChangeWOadj=mdl.Residuals.Raw;
mdl=fitglm(rawChangePR,PRpainpre);
rawChangePRadj=mdl.Residuals.Raw;
%% %% ====== PREDICTIONSß ===========

ALL=[rawChangeTKA,rawChangeTKAadj, ratioTKA,rawChangeWO,rawChangeWOadj,ratioWO, TKApainpre,WOpainpre, ROIs, genotype];

% Define the pattern
pattern = 'SC|CC';
% Use regexp to find indices where the numericTitles match the pattern
indices = find(~cellfun('isempty', regexp(numericTitles, pattern)));
numericTitles_sub=numericTitles(indices);


k =6;

if k==1
    varname = 'RawChange TKA';      
    ALL(isnan(ALL(:,k)),:)=[];
    input = ALL(:,7);
    predictorsCombined = ALL(:, [7, 9:end]);
    numericTitles_sub=['TKA pain pre',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
    target=ALL(:,k);
elseif k==2
    ALL(isnan(ALL(:,k)),:)=[];
    varname = 'RawChange TKA adj';
    input = ALL(:,7);
    predictorsCombined = ALL(:, [7, 9:end]);
    numericTitles_sub=['TKA pain pre',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
    target=ALL(:,k);
elseif k==3
    ALL(isnan(ALL(:,k)),:)=[];
    varname = 'Normalised Improvement TKA';
    input = ALL(:,7);
    predictorsCombined = ALL(:, [7, 9:end]);
    numericTitles_sub=['TKA pain pre',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
    target=ALL(:,k);
elseif k==4
    varname = 'RawChange WO';
    input = ALL(:,8);
    predictorsCombined = ALL(:,8:end);
    numericTitles_sub=['Pre-TKA,WOMAC (pain)',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
    target=ALL(:,k);
elseif k==5
    varname = 'RawChange WO adj';
    input = ALL(:,8);
    predictorsCombined = ALL(:,8:end);
    numericTitles_sub=['Pre-TKA,WOMAC (pain)',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
    target=ALL(:,k);
elseif k==6
    varname = 'Normalised Improvement WO';
    input = ALL(:,8);
    predictorsCombined = ALL(:,8:end);
    numericTitles_sub=['Pre-TKA,WOMAC (pain)',numericTitles_sub,numericTitles(find(~cellfun('isempty', regexp(numericTitles, 'Genotype'))))];
    target=ALL(:,k);
end


%%
rng(32); 

% Number of folds for cross-validation
numFolds = 15;

% Initialize result containers
allPredictionsSimpleLinear = [];
allTrueDataSimpleLinear = [];
allPredictionsWithPredictors = [];
allTrueDataWithPredictors = [];

% Loop for k-fold cross-validation
for fold = 1:numFolds
    % Create training and testing sets
    cv = cvpartition(height(input), 'KFold', numFolds);
    idxTrain = training(cv, fold);
    idxTest = test(cv, fold);

    % Simple Linear Regression
    X_train_simple = input(idxTrain);
    y_train_simple = target(idxTrain);

    X_test_simple = input(idxTest);
    y_test_simple = target(idxTest);

    % Standardize the features
    X_train_simple_scaled = zscore(X_train_simple);
    X_test_simple_scaled = zscore(X_test_simple);

    % Train a simple linear regression model
    mdl_simple = fitlm(X_train_simple_scaled, y_train_simple);

    % Make predictions on the test set
    predictions_simple = predict(mdl_simple, X_test_simple_scaled);

    % Store predictions and true data for simple linear regression
    allPredictionsSimpleLinear = [allPredictionsSimpleLinear; predictions_simple];
    allTrueDataSimpleLinear = [allTrueDataSimpleLinear; y_test_simple];

    % Regression with Predictors Combined
    X_train_combined = predictorsCombined(idxTrain, :);
    y_train_combined = target(idxTrain);

    X_test_combined = predictorsCombined(idxTest, :);
    y_test_combined = target(idxTest);

    % Standardize the features
    X_train_combined_scaled = zscore(X_train_combined);
    X_test_combined_scaled = zscore(X_test_combined);

    % Feature selection
    numSelectedFeatures = 60;
    selectedFeaturesidx = sequentialfs(@critfun, X_train_combined_scaled, y_train_combined, 'cv', 'none', 'Nfeatures', numSelectedFeatures);
    selectedFeatures = X_train_combined_scaled(:, selectedFeaturesidx);

    % Train a regression model with predictors combined
    mdl_combined = fitrsvm(selectedFeatures, y_train_combined, 'Standardize', true);

    % Make predictions on the test set using the selected features
    predictions_combined = predict(mdl_combined, X_test_combined_scaled(:, selectedFeaturesidx));

    % Store predictions and true data for regression with predictors combined
    allPredictionsWithPredictors = [allPredictionsWithPredictors; predictions_combined];
    allTrueDataWithPredictors = [allTrueDataWithPredictors; y_test_combined];
    fprintf("HII I AM HERE%d\n",fold);
end

% Plot for Simple Linear Regression
figure(1);
subplot(1, 2, 1);
[rho1, p1] = PlotSimpleCorrelationWithRegression(allTrueDataSimpleLinear, allPredictionsSimpleLinear, 30, 'b');
title({"Model: TkaPre vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho1, p1)});
ylabel('Predicted');
xlabel('True');
hold off;

% Plot for Regression with Predictors Combined
subplot(1, 2, 2);
[rho2, p2] = PlotSimpleCorrelationWithRegression(allTrueDataWithPredictors, allPredictionsWithPredictors, 30, 'b');
title({"Model: [TkaPre, ROIs, geno] vs", sprintf("%s", varname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)});
ylabel('Predicted');
xlabel('True');
hold off;


%%
function crit = critfun(x, y)
    mdl = fitlm(x, y);
    crit = mdl.RMSE;
end

