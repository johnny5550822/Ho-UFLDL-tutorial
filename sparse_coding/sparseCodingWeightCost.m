function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures,  patches, gamma, lambda, epsilon, groupMatrix)

% NOTICE: there is no base involved here. To have base, simply add
% additional row to W and col to W (for base)

%sparseCodingWeightCost - given the features in featureMatrix, 
%                         computes the cost and gradient with respect to
%                         the weights, given in weightMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.
m = size(patches,2);

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
%     size(featureMatrix)
%     numFeatures
%     numExamples
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------    

    %calculate cost
    normal_cost = 1/m*sum(sum((weightMatrix * featureMatrix - patches).^2));    %64x6
    sparsity_cost = lambda/m* sum(sum(sqrt(groupMatrix * (featureMatrix.*featureMatrix) +epsilon)));
    weight_decay = gamma * (sum(sum(weightMatrix.^2)));
    cost = (normal_cost + sparsity_cost + weight_decay);

    %calculate gradient
    grad = 2/m* (weightMatrix * featureMatrix - patches) * featureMatrix' + ...
        2 * gamma * weightMatrix;   %64x5
    grad = grad(:);
    
end