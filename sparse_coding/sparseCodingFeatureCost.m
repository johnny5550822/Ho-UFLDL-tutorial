function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingFeatureCost - given the weights in weightMatrix,
%                          computes the cost and gradient with respect to
%                          the features, given in featureMatrix
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
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   features given in featureMatrix.     
    %   You may wish to write the non-topographic version, ignoring
    %   the grouping matrix groupMatrix first, and extend the 
    %   non-topographic version to the topographic version later.
    % -------------------- YOUR CODE HERE --------------------
    
    % #######non-topographic version
%     %calculate cost
%     normal_cost = sum(sum((weightMatrix * featureMatrix - patches).^2));    %64x6
%     sparsity_cost = lambda * sum(sum(sqrt(featureMatrix.^2 +epsilon)));
%     weight_decay = gamma * (sum(sum(weightMatrix.^2)));
%     cost = normal_cost + sparsity_cost + weight_decay;
%     
%     %calculate gradient
%     grad = 2 * weightMatrix' * (weightMatrix * featureMatrix - patches) + ...
%         lambda * (featureMatrix./sqrt((featureMatrix.^2)+epsilon));   %5x6
%     grad = grad(:);
        
%     % #######topographic version    
    normal_cost = 1/m*sum(sum((weightMatrix * featureMatrix - patches).^2));    %64x6
    sparsity_cost = lambda/m * sum(sum(sqrt(groupMatrix * (featureMatrix.*featureMatrix) +epsilon)));
    % The penalty term(sparsity_cost) in the note is not written correctly.
    % Edited here
    weight_decay = gamma * (sum(sum(weightMatrix.^2)));
    cost = (normal_cost + sparsity_cost + weight_decay);
    
    %calculate gradient
    normal_grad = 2/m * weightMatrix' * (weightMatrix * featureMatrix - patches); %5x6
    l1_penalty_grad = lambda/m * groupMatrix' * (1./sqrt(groupMatrix*(featureMatrix.*featureMatrix)+ ...
        epsilon)) .* featureMatrix;   %5x6; This one has to be derived from derivative of cost function
    grad = (normal_grad + l1_penalty_grad);
    grad = grad(:);
end