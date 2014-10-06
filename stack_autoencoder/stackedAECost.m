%???????????There are some problems in this code????
function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);
%size = 10x20

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
%e.g. stack{1}.w =size: 20x784
%e.g. stack{1}.b =size: 20x1

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));   %10(10 digit)x20(20 variable; without base)
%size(softmaxTheta)

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    % for d = 1 (1st layer), stackgrad{1}.w = [20x784], .b=20x1
    % for d = 2 (2nd layer), stackgrad{2}.w = [20x20], .b=20x1
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end
cost = 0; % You need to compute this

% You might find these variables useful
%size(data); %784x11
M = size(data, 2);  %11
groundTruth = full(sparse(labels, 1:M, 1)); % 10x11

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%----------------------------Step1: Forward propagation
%Pre-process; combine both base and the regular parameters
W_b1 = [stack{1}.b stack{1}.w]; %20x785
W_b2 = [stack{2}.b stack{2}.w]; %20x21
%size(softmaxTheta)  %10x20
data = data';
data = [ones(size(data, 1), 1) data];   %11x785
%########Propagation#########
%1st layer(input) to 2nd layer(1st hidden layer) activation
z_2 = W_b1*data'; %20*11
a_2 = sigmoid(z_2); %20*11
a_2 = [ones(1,size(data,1)) ; a_2]; %21x11

%2nd layer to 3rd layer(2nd hidden layer) activation
z_3 = W_b2*a_2; %20x11
a_3 = sigmoid(z_3); %20x11(20 unit;11 examples)
% a_3 = [ones(1,size(data,1)) ; a_3]; %21(activiation)x11(sample) <-- for softmax; may not
% %necessary

%3rd layer to softmax
z_4 = softmaxTheta*a_3; %10x11
%size(z_4)
% size(softmaxTheta) %10x20
% size(a_3)

% #######calculate cost (CORRECT)
hypothesis = calculate_hypothesis(softmaxTheta,a_3);  %10x11
each_k = groundTruth.*log(hypothesis); %the inner summation of the cost with respect to k
regularization = lambda/2 * sum(sum(softmaxTheta.^2));
simple_cost = -1/M*sum(sum(each_k,1));
cost = simple_cost+regularization;

% #######calculate thetagrad for softmax
difference = groundTruth - hypothesis;  %10x11
%tri_thetagrad = softmaxTheta(:,2:end)' * difference; %20x11; 
simple_thetagrad = -1/M*(difference * a_3');    %10x20
regularization_grad = lambda * softmaxTheta; 
thetagrad = simple_thetagrad + regularization_grad; %10(digit)x20(parameters)

%------------------------------Step2: backpropagation
%size(softmaxTheta' * difference); %20x11
%size(sigmoidGradient(z_4))  %10x11
delta_last = -(softmaxTheta' *difference).* sigmoidGradient(z_3); %10(parameters)x11
        %<-- at layer 3
delta_2 = (W_b2(:,2:end)'*delta_last).*sigmoidGradient(z_2); %20x11
        %<--- at layer 2 

tri_2 = delta_last*a_2';   %20x21  
tri_1 = delta_2*data;   %20x785

%-----------------------------Step3: Update
% DO not regularized the bias term
Theta1_grad = 1/M*tri_1;  %size=20*785
Theta2_grad = 1/M*tri_2;  %size=20*21

% no regularization on the bias term
% Theta1_grad(:,1) = Theta1_grad(:,1) - lambda*W_b1(:,1);
% Theta2_grad(:,1) = Theta2_grad(:,1) - lambda*W_b2(:,1);

%Assign value
stackgrad{1}.w = Theta1_grad(:,2:end);
stackgrad{2}.w = Theta2_grad(:,2:end);
stackgrad{1}.b = Theta1_grad(:,1);
stackgrad{2}.b = Theta2_grad(:,1);
softmaxThetaGrad = thetagrad;

% -------------------------------------------------------------------------


%%%reference
% %------------------------------Step2: backpropagation
% %size(softmaxTheta' * difference); %20x11
% %size(sigmoidGradient(z_4))  %10x11
% delta_last = -(softmaxTheta' *difference).* sigmoidGradient(z_3); %10(parameters)x11
%         %<-- for layer W2
% delta_2 = (W_b2(:,2:end)'*delta_last).*sigmoidGradient(z_2); %20x11
%         %<---
% %delta_2 = (W_b2(:,2:end)'*delta_3).*sigmoidGradient(z_2); %20x11
% 
% tri_3 = delta_last*a_3'; %10x21
% tri_2 = delta_3*a_2';   %20x21  
% tri_1 = delta_2*data;   %20x785
% 
% %-----------------------------Step3: Update
% % DO not regularized the bias term
% Theta1_grad = 1/M*tri_1 + lambda*W_b1; %size=20*785
% Theta2_grad = 1/M*tri_2 + lambda*W_b2; %size=20*21
% Theta3_grad = 1/M*tri_3 + lambda*softmaxTheta; %size=10*21
% 
% % no regularization on the bias term
% Theta1_grad(:,1) = Theta1_grad(:,1) - lambda*W_b1(:,1);
% Theta2_grad(:,1) = Theta2_grad(:,1) - lambda*W_b2(:,1);
% Theta3_grad(:,1) = Theta3_grad(:,1) - lambda*softmaxTheta(:,1);
% 
% %Assign value
% stackgrad{1}.w = Theta1_grad(:,2:end);
% stackgrad{2}.w = Theta2_grad(:,2:end);
% stackgrad{1}.b = Theta1_grad(:,1);
% stackgrad{2}.b = Theta2_grad(:,1);
% softmaxThetaGrad = thetagrad;




%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% % You might find this useful
% function sigm = sigmoid(x)
%     sigm = 1 ./ (1 + exp(-x));
% end
function hypothesis = calculate_hypothesis(theta,data)
    %calculate the denominator of hypothesis
    exp_power = theta * data; %10x8  8x100 -->10x100
    % subract the largest value to avoid possible overflow when exponential
    % is applied.
    exp_power = bsxfun(@minus, exp_power, max(exp_power,[],1)); 
    denominator = sum(exp(exp_power),1) ;   %size=1x100
    
    %calculate hypothesis (must be 10x100<--for each exponential of 10
    %classes, and 100samples)
    hypothesis = exp(exp_power)./repmat(denominator,size(exp_power,1),1); %10x100
end