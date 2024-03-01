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
    variables.Genotype_1_GG_, variables.Sex_1_F_, variables.ROIs]);
% Now, dynamically construct featureNames based on the variables you've included in ALL_DATA
pNamesBase = {'WpainImpr', 'PreWpain', 'Genotype', 'Sex'};
variables.modelname='PrePain + Genotype + Sex + ROIs';
variables.targetname=pNamesBase{1};

roiFeatureNames = variables.ROIstitles;
allpNames = [pNamesBase, roiFeatureNames];

% Assuming first column is the target for analysis
[target, input, featureNames] = prepareData(ALL_DATA, allpNames);

% Leave-One-Out Cross-Validation with Feature Selection
[selectedFeaturesSTORE, predictions, predictedTarget, STORE] = leaveOneOutCV(input, target, featureNames);
%
% Plotting results
plotResults(variables, target, predictions, predictedTarget, STORE, featureNames);



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


function [selectedFeaturesSTORE, predictions, predictedTarget, STORE] = leaveOneOutCV(input, target, featureNames)
    numFolds = size(input, 1);
    STORE = zeros(length(featureNames), numFolds); % Store selected features for each fold
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
        [selectedFeatures, ~] = featureSelectionSVM(inputTrain, targetTrain, numSelectedFeatures);

        % Store selected features
        STORE(:, fold) = selectedFeatures;
        selectedInputTrain = inputTrain(:, selectedFeatures==1);
        selectedInputTest = inputTest(:, selectedFeatures==1);
        % SVM prediction
        combinedModel = fitrsvm(selectedInputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
        % Make predictions on the test data
        predictedTarget(testIndex) = predict(combinedModel, selectedInputTest);

        % Fit linear model with selected predictors for input vs target
        mdl1 = fitlm(inputTrain(:, 1), targetTrain);
        predictions(testIndex) = predict(mdl1, inputTest(1));

    
    end

    selectedFeaturesSTORE = sum(STORE, 2); % Sum selected features across folds
end

function [selectedFeatures, model] = featureSelectionSVM(inputTrain, targetTrain, numSelectedFeatures)
    % Implement RFE or other feature selection method here
    % Placeholder for actual implementation
    model = fitrsvm(inputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
    featureWeights = abs(model.Beta);
    [~, featureIdx] = sort(featureWeights, 'descend');
    selectedFeaturesIdx = featureIdx(1:numSelectedFeatures);
    selectedFeatures = zeros(size(inputTrain, 2), 1);
    selectedFeatures(selectedFeaturesIdx) = 1;
end

%%
function plotResults(variables, target, predictions, predictedTarget, STORE, featureNames)
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
    
    [sortedFreq, sortedTitleStrings] = sortfreq(STORE,featureNames);

    % Plot the histogram using the histogram function
    subplot(2,1,2)
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

    hold off

    figure(2)
    [rho2, p2] = PlotSimpleCorrelationWithRegression(target,  predictions, 30, 'b');
    title({sprintf('Model: PrePain'), sprintf("Rho: %.2f; p: %.2f", rho2, p2)},'FontSize',18);
    ylabel('Pred. Improvement','FontSize',18);
    xlabel('True Improvement','FontSize',18);
    xlim([0,1.3])
    ylim([0,1.3])

    hold off
end


function [sortedFreq, sortedTitleStrings] = sortfreq(STORE,featureNames)
% Sum the frequencies across folds for selected features
    freq = sum(STORE, 2);
    % Convert freq to a cell array
    freqCell = num2cell(freq);
    % Concatenate freqCell and title
    combinedData = [freqCell, featureNames'];

    % Sort based on the first column (freq) in descending order
    sortedData = sortrows(combinedData, -1);
    
    % Extract the sorted freq and title
    sortedFreq = cell2mat(sortedData(:, 1));
    sortedTitle = sortedData(:, 2);
    % Plot the histogram using the histogram function
    % Plot the histogram
    % Convert the cell array of titles to a cell array of strings
    sortedTitleStrings = cellfun(@str2mat, sortedTitle, 'UniformOutput', false);
end