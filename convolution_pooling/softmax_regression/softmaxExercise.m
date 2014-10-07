%% CS294A/CS294W Softmax Exercise 

%!!!!!!!!!Remember to mind the base(1) in softmax, change inputsize to
%imagedimension+1; add additional row(1s) to image.

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  softmax exercise. You will need to write the softmax cost function 
%  in softmaxCost.m and the softmax prediction function in softmaxPred.m. 
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%  (However, you may be required to do so in later exercises)

%%======================================================================
%% STEP 0: Initialise constants and parameters
%
%  Here we define and initialise some constants which allow your code
%  to be used more generally on any arbitrary input. 
%  We also initialise some parameters used for tuning the model.
clear all;
clc;
inputSize = 28 * 28+1; % Size of input vector (MNIST images are 28x28) + base term
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data
%
%  In this section, we load the input and output data.
%  For softmax regression on MNIST pixels, 
%  the input data is the images, and 
%  the output data is the labels.
%

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
clc;
images = loadMNISTImages('mnist/train-images.idx3-ubyte');
labels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;
%!!!!!!!for base
inputData = [ones(1,size(inputData,2));inputData];

%size(inputData)
% For debugging purposes, you may wish to reduce the size of the input data
% in order to speed up gradient checking. 
% Here, we create synthetic dataset using random data for testing

DEBUG = false; % Set DEBUG to true when debugging.
DEBUG_custom = false;   %my custom debug with similar training set
if DEBUG
    inputSize = 8;
    inputData = randn(8, 100);
    labels = randi(10, 100, 1);
end
if DEBUG_custom
    inputSize = 8+1;    %for base term
    num_training = 1000;
    inputData = inputData(1:inputSize-1,1:num_training);
    labels = randi(10, num_training, 1);
end

% Randomly initialise theta
%theta = 0.005 * randn(numClasses * inputSize, 1);
theta = 0.005 * randn(numClasses * (inputSize), 1); 


%%======================================================================
%% STEP 2: Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 
clc;
[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
                                     
%%======================================================================
%% STEP 3: Gradient checking
%
%  As with any learning algorithm, you should always check that your
%  gradients are correct before learning the parameters.
% 
clc;
if DEBUG || DEBUG_custom
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);
%     numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
%                                     inputSize+1, lambda, ...
%                                     [ones(1,size(inputData,2));inputData], labels),...
%                                     [zeros(size(theta,1),1) theta]);

    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 

    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end

%%======================================================================
%% STEP 4: Learning parameters
%
%  Once you have verified that your gradients are correct, 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).
clc;
options.maxIter = 100;
addpath minFunc/
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing
%
%  You should now test your model against the test images.
%  To do this, you will first need to write softmaxPredict
%  (in softmaxPredict.m), which should return predictions
%  given a softmax model and the input data.
clc;
images = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
labels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;
%!!!!!!!!!for base
inputData = [ones(1,size(inputData,2));inputData];

% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
% If your values are too low (accuracy less than 0.91), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
