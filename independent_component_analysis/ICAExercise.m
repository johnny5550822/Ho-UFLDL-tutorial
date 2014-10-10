%% CS294A/CS294W Independent Component Analysis (ICA) Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  ICA exercise. In this exercise, you will need to modify
%  orthonormalICACost.m and a small part of this file, ICAExercise.m.

%%======================================================================
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.
clear all;
clc;
addpath linear_autoencoder
addpath sparse_autoencoder
numPatches = 20000;
numFeatures = 121;
imageChannels = 3;
patchDim = 8;
visibleSize = patchDim * patchDim * imageChannels;

outputDir = '.';
epsilon = 1e-6; % L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon)

%%======================================================================
%% STEP 1: Sample patches

patches = load('stlSampledPatches.mat');
patches = patches.patches(:, 1:numPatches);
displayColorNetwork(patches(:, 1:100));

%%======================================================================
%% STEP 2: ZCA whiten patches
%  In this step, we ZCA whiten the sampled patches. This is necessary for
%  orthonormal ICA to work.

patches = patches / 255;
meanPatch = mean(patches, 2);
patches = bsxfun(@minus, patches, meanPatch);

sigma = patches * patches';
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s))) * u';
patches = ZCAWhite * patches;

%%======================================================================
%% STEP 3: ICA cost functions
%  Implement the cost function for orthornomal ICA (you don't have to 
%  enforce the orthonormality constraint in the cost function) 
%  in the function orthonormalICACost in orthonormalICACost.m.
%  Once you have implemented the function, check the gradient.
clc;
% Use less features and smaller patches for speed
numFeatures = 5;
patches = patches(1:3, 1:5);    %3x5(number of patches)
visibleSize = 3;
numPatches = 5;

weightMatrix = rand(numFeatures, visibleSize);  %5x3

[cost, grad] = orthonormalICACost(weightMatrix, visibleSize, numFeatures, patches, epsilon);
numGrad = computeNumericalGradient( @(x) orthonormalICACost(x, visibleSize, numFeatures, patches, epsilon), weightMatrix(:) );
% size(grad)
% size(numGrad)
% Uncomment to display the numeric and analytic gradients side-by-side
% disp([numGrad grad]); 
diff = norm(numGrad-grad)/norm(numGrad+grad);
fprintf('Orthonormal ICA difference: %g\n', diff);
assert(diff < 1e-7, 'Difference too large. Check your analytic gradients.');

fprintf('Congratulations! Your gradients seem okay.\n');

%%======================================================================
%% STEP 4: Optimization for orthonormal ICA
%  Optimize for the orthonormal ICA objective, enforcing the orthonormality
%  constraint. Code has been provided to do the gradient descent with a
%  backtracking line search using the orthonormalICACost function 
%  (for more information about backtracking line search, you can read the 
%  appendix of the exercise).
%
%  However, you will need to write code to enforce the orthonormality 
%  constraint by projecting weightMatrix back into the space of matrices 
%  satisfying WW^T  = I.
%
%  Once you are done, you can run the code. 10000 iterations of gradient
%  descent will take around 2 hours, and only a few bases will be
%  completely learned within 10000 iterations. This highlights one of the
%  weaknesses of orthonormal ICA - it is difficult to optimize for the
%  objective function while enforcing the orthonormality constraint - 
%  convergence using gradient descent and projection is very slow.

weightMatrix = rand(numFeatures, visibleSize);

[cost, grad] = orthonormalICACost(weightMatrix(:), visibleSize, numFeatures, patches, epsilon);

fprintf('%11s%16s%10s\n','Iteration','Cost','t');

startTime = tic();

% Initialize some parameters for the backtracking line search
alpha = 0.5;
t = 0.02;
lastCost = 1e40;

% Do 10000 iterations of gradient descent
for iteration = 1:10000
                       
    grad = reshape(grad, size(weightMatrix));
    newCost = Inf;        
    linearDelta = sum(sum(grad .* grad));
    
    % Perform the backtracking line search
    while 1
        considerWeightMatrix = weightMatrix - alpha * grad;
        % -------------------- YOUR CODE HERE --------------------
        % Instructions:
        %   Write code to project considerWeightMatrix back into the space
        %   of matrices satisfying WW^T = I.
        %   
        %   Once that is done, verify that your projection is correct by 
        %   using the checking code below. After you have verified your
        %   code, comment out the checking code before running the
        %   optimization.
        
        % Project considerWeightMatrix such that it satisfies WW^T = I
        %cwm = considerWeightMatrix;
        considerWeightMatrix = (considerWeightMatrix*considerWeightMatrix')^(-1/2) * considerWeightMatrix;

        % Verify that the projection is correct; if you test with sample
        % images size, i.e. numPatches = 5, it will cause error here
%         temp = considerWeightMatrix * considerWeightMatrix';
%         temp = temp - eye(numFeatures);
%         assert(sum(temp(:).^2) < 1e-23, 'considerWeightMatrix does not satisfy WW^T = I. Check your projection again');
%         error('Projection seems okay. Comment out verification code before running optimization.');
%         
        % -------------------- YOUR CODE HERE --------------------                                        

        [newCost, newGrad] = orthonormalICACost(considerWeightMatrix(:), visibleSize, numFeatures, patches, epsilon);
        if newCost > lastCost - alpha * t * linearDelta
            t = 0.9 * t;
        else
            break;
        end
    end
   
    lastCost = newCost;
    weightMatrix = considerWeightMatrix;
    
    fprintf('  %9d  %14.6f  %8.7g\n', iteration, newCost, t);
    
    t = 1.1 * t;
    
    cost = newCost;
    grad = newGrad;
           
    % Visualize the learned bases as we go along    
    if mod(iteration, 1000) == 0
        duration = toc(startTime);
        % Visualize the learned bases over time in different figures so 
        % we can get a feel for the slow rate of convergence
        figure(floor(iteration /  1000));
        displayColorNetwork(weightMatrix'); 
    end
                   
end

% Visualize the learned bases
displayColorNetwork(weightMatrix');