% Clear workspace and command window
clear all;
clc;
rng(42); % Ensure reproducibility

% Set and add paths
dataFilePath = '/Users/e410377/Desktop/KOAfeatures/FINALWOMAC.xlsx';
utilityPath = '/Users/e410377/Desktop/PETAnalysisPaper/utility';
addpath(genpath(utilityPath));

% Import and preprocess data
[data, titles] = importExcelData(dataFilePath);
variables = extractVariables(data, titles);
variables = extractROIs(variables);

% Perform computations
variables = computeMetrics(variables);

% Normalization and preparation for analysis
ALL_DATA = normalizeData([variables.ratio_Pre_TKA_WOMAC_pain_, variables.Pre_TKA_WOMAC_pain_, ...
    variables.Genotype0, variables.Genotype1, variables.Sex0, variables.Sex1, variables.ROIs]);
% Now, dynamically construct featureNames based on the variables you've included in ALL_DATA
pNamesBase = {'WpainImpr', 'PreWpain', 'HAB','MAB', 'F', 'M'};
variables.modelname='PrePain + Genotype + Sex + ROIs';
variables.targetname=pNamesBase{1};

roiFeatureNames = variables.ROIstitles;
allpNames = [pNamesBase, roiFeatureNames];

% Assuming first column is the target for analysis
[target, input, featureNames] = prepareData(ALL_DATA, allpNames);

% Leave-One-Out Cross-Validation with Feature Selection
[selectedFeaturesSTORE, predictions, predictedTarget, STORE, WEIGHTS] = leaveOneOutCV(input, target, featureNames,'SVM');
%
% Plotting results
plotResults(variables, target, predictions, predictedTarget, STORE, WEIGHTS, featureNames);



%% ================ Function Definitions  ================

function [data, titles] = importExcelData(filePath)
    importedData = importdata(filePath);
    data = importedData.data;
    titles = importedData.textdata(1, 2:end);
end

function variables = extractVariables(data, titles)
    variables.numericData = data;
    variables.titles = titles;
    % Extract specific variables
    extract = @(name) data(:, strcmp(titles, name));
    varNames = {'Age (pre)', 'Genotype (1=GG)', 'Sex (1=F)', 'TKA pain pre', 'TKA pain post', ...
                'Pre-TKA,WOMAC (pain)', 'Pre-TKA,WOMAC (physical function)', 'Pre-TKA,WOMAC (stiffness)', ...
                '1yr POST-TKA, Womac (pain)', '1yr POST-TKA,Womac (phys func)', '1yr POST-TKA,Womac (stiffness)', ...
                'PRE-TKA, Promis (pain intensity)'};
    for varName = varNames
        cleanedName = matlab.lang.makeValidName(varName{1});
        variables.(cleanedName) = extract(varName{1});
    end
    % Assuming genotype is your original array
    variables.Genotype0 = variables.Genotype_1_GG_ == 1; % This will be 1 (true) where genotype is 0, and 0 (false) otherwise
    variables.Genotype1 = variables.Genotype_1_GG_ == 2; % This will be 1 (true) where genotype is 1, and 0 (false) otherwise
    
    
    % Assuming genotype is your original array
    variables.Sex0 = variables.Sex_1_F_ == 1; % This will be 1 (true) where genotype is 0, and 0 (false) otherwise
    variables.Sex1 = variables.Sex_1_F_ == 2; % This will be 1 (true) where genotype is 1, and 0 (false) otherwise
end

function variables = computeMetrics(variables)
    eps = 1e-10; % For numerical stability
    computeRatio = @(pre, post) (pre - post) ./ (pre + post + eps);
    
    % Define variable pairs for ratio computation
    pairs = {'TKAPainPre', 'TKAPainPost'; ...
             'Pre_TKA_WOMAC_pain_', 'x1yrPOST_TKA_Womac_pain_'; ...
             'Pre_TKA_WOMAC_physicalFunction_', 'x1yrPOST_TKA_Womac_physFunc_'; ...
             'Pre_TKA_WOMAC_stiffness_', 'x1yrPOST_TKA_Womac_stiffness_'};
    
    for i = 1:size(pairs, 1)
        preVar = variables.(pairs{i, 1});
        postVar = variables.(pairs{i, 2});
        ratioName = ['ratio_' pairs{i, 1}];
        variables.(ratioName) = computeRatio(preVar, postVar);
    end
end

function variables = extractROIs(variables)
    % Extract ROIs using a pattern match for titles starting with 'SC', 'CC', or 'PG'
    pattern = '^(SC|CC|PG)';
    roiIndices = find(~cellfun('isempty', regexp(variables.titles, pattern)));
    variables.ROIs = variables.numericData(:, roiIndices);
    variables.ROIstitles=variables.titles(:,roiIndices);
end

function out = normalizeData(input)
    [rows, cols] = size(input);
    out = zeros(rows, cols);
    for col = 1:cols
        % Normalize each column separately
        colData = input(:, col);
        out(:, col) = (colData - min(colData)) ./ (max(colData) - min(colData));
    end
end

function [target, input, featureNames] = prepareData(ALL_DATA, titles)
    target = ALL_DATA(:, 1); % Assuming the first column is the target
    input = ALL_DATA(:, 2:end); % The rest are features
    % Adjust featureNames based on your actual data structure
    featureNames = titles(2:end); % Example adjustment
end


function [selectedFeaturesSTORE, predictions, predictedTarget, STORE, WEIGHTS] = leaveOneOutCV(input, target, featureNames, method)
    numFolds = size(input, 1);
    STORE = zeros(length(featureNames), numFolds); % Store selected features for each fold
    WEIGHTS = zeros(length(featureNames), numFolds); % Store selected features for each fold
    predictions = zeros(size(target));
    predictedTarget = zeros(size(target));
    numSelectedFeatures = round(0.21 * size(input, 2)); % Adjust as needed

    for fold = 1:numFolds
        fprintf('Processing fold %d/%d...\n', fold, numFolds);
        testIndex = fold;
        trainIndex = setdiff(1:numFolds, testIndex);

        % Splitting data
        inputTrain = input(trainIndex, :);
        targetTrain = target(trainIndex);
        inputTest = input(testIndex, :);

        % Feature selection and SVM training
        [selectedFeatures, model, coeff, intercept] = performFeatureSelection(inputTrain, targetTrain, method, numSelectedFeatures);

        if strcmp(method, 'ElasticNet')

            prediction = inputTest * coeff + intercept; % Adjust for Elastic Net
            predictedTarget(testIndex) = prediction;
            WEIGHTS = [WEIGHTS, coeff]; % Append the coefficients
            STORE(:, fold) = selectedFeatures;
        elseif strcmp(method, 'SVM')
            selectedInputTrain = inputTrain(:, selectedFeatures==1);
            selectedInputTest = inputTest(:, selectedFeatures==1);
            combinedModel = fitrsvm(selectedInputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
            predictedTarget(testIndex) = predict(combinedModel, selectedInputTest);
            WEIGHTS(:, fold) = coeff;
            STORE(:, fold) = selectedFeatures;
        end

        % Fit linear model with selected predictors for input vs target
        mdl1 = fitlm(inputTrain(:, 1), targetTrain);
        predictions(testIndex) = predict(mdl1, inputTest(1));
        
        
        
    end

    selectedFeaturesSTORE = sum(STORE, 2); % Sum selected features across folds
end

function [selectedFeatures, model, coeff,intercept] = performFeatureSelection(inputTrain, targetTrain, method, numSelectedFeatures)
    if strcmp(method, 'ElasticNet')
        [B, FitInfo] = lasso(inputTrain, targetTrain, 'CV', 10); % Use 10-fold CV to choose lambda
        bestLambda = FitInfo.LambdaMinMSE;
        coeff = B(:, FitInfo.IndexMinMSE);
        selectedFeatures = coeff ~= 0; % Indicator vector for selected features
        model = []; % Elastic Net does not return a model object in the same way as SVM
        intercept = FitInfo.Intercept(FitInfo.IndexMinMSE);
    elseif strcmp(method, 'SVM')
        % Placeholder for SVM feature selection, adjust as needed
        % This example uses all features and fits an SVM, real feature selection for SVM might require a different approach
        model = fitrsvm(inputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
        featureWeights = abs(model.Beta);
        [~, featureIdx] = sort(featureWeights, 'descend');
        selectedFeaturesIdx = featureIdx(1:numSelectedFeatures);
        selectedFeatures = zeros(size(inputTrain, 2), 1);
        selectedFeatures(selectedFeaturesIdx) = 1;
        coeff = model.Beta; % Coefficients (weights) of the SVM model
        intercept = [];
    else
        error('Unsupported method. Choose either "ElasticNet" or "SVM".');
    end
end


%%
function plotResults(variables, target, predictions, predictedTarget, STORE, WEIGHTS,  featureNames)
    % Placeholder for result plotting
    % Access variables to plot graphs or results here
    subplot(2,3,2)
    [rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictedTarget, 30, 'b');
    title({sprintf('Model:%s',variables.modelname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)},'FontSize',18);
    ylabel('Pred. Improvement','FontSize',18);
    xlabel('True Improvement','FontSize',18);
    xlim([0,1.3])
    ylim([0,1.3])
    hold off
    
    [sortedFreq, sortedTitleStrings, sortedWeights] = sortfreq(STORE, WEIGHTS, featureNames);

    % Normalize the magnitude of sortedWeights to [0, 1] for transparency
    maxWeight = max(abs(sortedWeights));
    normalizedIntensity = abs(sortedWeights) / maxWeight;
    
    % Plot the histogram using colored bars with transparency
    subplot(2,1,2);
    % Plot with adjusted transparency
    for i = 1:length(sortedFreq)
        barColor = [1, 0, 0]; % Default to red, adjust as needed based on directionality
        if sortedWeights(i) < 0
            barColor = [0, 0, 1]; % Adjust to blue for negative weights
        end
        bar(i, sortedFreq(i), 'FaceColor', barColor, 'EdgeColor', 'none', 'FaceAlpha', normalizedIntensity(i));
        hold on;
    end
    hold off; % Release the plot for further commands
    
    % Set the x-axis labels to be the sorted titles
    xticks(1:length(sortedFreq));
    xticklabels(sortedTitleStrings);
    xtickangle(45);
    
    % Set axis labels and title
    xlabel('Features');
    ylabel('Frequency');
    set(gcf, 'Color', 'w');
    set(gca, 'FontSize', 15);
    grid on;
        


    figure(2)
    [rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictions, 30, 'b');
    title({sprintf('Model: PrePain'), sprintf("Rho: %.2f; p: %.2f", rho2, p2)},'FontSize',18);
    ylabel('Pred. Improvement','FontSize',18);
    xlabel('True Improvement','FontSize',18);
    xlim([0,1.3])
    ylim([0,1.3])

    hold off
end


function [sortedFreq, sortedTitleStrings, sortedWeights] = sortfreq(STORE, WEIGHTS, featureNames)
% Sum the frequencies across folds for selected features
    freq = sum(STORE, 2);
    %%
    sWEIGHTS = scaledown(WEIGHTS);
    % Normalize weights row-wise for consistency in directionality
    normalizedWeights = normalizeRows(sWEIGHTS);
    
    % Sum normalized weights across folds
    summedNormalizedWeights = sum(normalizedWeights, 2);
    summedNormalizedWeights = scaledown(summedNormalizedWeights);

    % Sum weights for each feature across folds
    %summedWeights = sum(sWEIGHTS, 2);

    % Concatenate freqCell and title
    combinedData = [num2cell(freq), featureNames', num2cell(summedNormalizedWeights)];

    % Sort based on the first column (freq) in descending order
    sortedData = sortrows(combinedData, -1);
    
    % Extract the sorted freq and title
    sortedFreq = cell2mat(sortedData(:, 1));
    sortedWeights = cell2mat(sortedData(:, 3));

    sortedTitle = sortedData(:, 2);
    % Plot the histogram using the histogram function
    % Plot the histogram
    % Convert the cell array of titles to a cell array of strings
    sortedTitleStrings = cellfun(@str2mat, sortedTitle, 'UniformOutput', false);
end

function scaledWeights = scaledown(WEIGHTS)
    % Initialize scaledWeights with the same size as WEIGHTS
    scaledWeights = zeros(size(WEIGHTS));
    
    % Iterate over each fold (column)
    for col = 1:size(WEIGHTS, 2)
        % Get the maximum absolute weight for the current column
        maxWeight = max(abs(WEIGHTS(:, col)));
        
        % Scale down the weights for the current column
        if maxWeight == 0
            % Avoid division by zero if maxWeight is 0
            scaledWeights(:, col) = WEIGHTS(:, col);
        else
            scaledWeights(:, col) = WEIGHTS(:, col) / maxWeight;
        end
    end
end

function normalizedWeights = normalizeRows(WEIGHTS)
    % Preallocate the normalizedWeights matrix
    normalizedWeights = zeros(size(WEIGHTS));
    
    % Iterate through each row (feature)
    for i = 1:size(WEIGHTS, 1)
        rowWeights = WEIGHTS(i, :);
        maxAbsWeight = max(abs(rowWeights));
        
        % Avoid division by zero
        if maxAbsWeight == 0
            normalizedWeights(i, :) = rowWeights;
        else
            normalizedWeights(i, :) = rowWeights / maxAbsWeight;
        end
    end
end
