function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------     
    W=weightMatrix;
    m = size(patches,2);    % number of patches
    inside = sqrt((W*patches).^2+epsilon);
    
    %calculate cost
    cost = 1/m*sum(inside(:));
    
    %calculate grad
    grad = 1/m*((W*patches)./inside)*patches';
    
    %unroll
    grad = grad(:);
end

