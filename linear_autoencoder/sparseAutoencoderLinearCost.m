function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
%!!!!!!!!!!Copy from sparseAutoencoderCost.m; some comments are from the
%old .m
                                                        
                                                        
% visibleSize: the number of input units (probably 64)
% hiddenSize: the number of hidden units (probably 25)
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention of the lecture notes.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
grad = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
%
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
%

%Pre-process; combine both base and the regular parameters
W_b1 = [b1 W1]; %25*65
W_b2 = [b2 W2]; %64*26

% #############Based on exercise from Coursea, Dr. Andrew Ng############
data = data'; %transpose the data so that it can be used by the code in Coursea
% Setup some useful variables
m = size(data, 1);
%INITIAL STEP
data = [ones(size(data, 1), 1) data];


%##############################Calculate cost##########
%Perform feed-forward propagation (vectorized)
% loop each digit

%------------------------Step 1: Forward propagation
%1st layer to 2nd(hidden) layer activation
z_2 = W_b1*data'; %25*10000
a_2 = sigmoid(z_2); %25*10000
p_optimal = 1/m*sum(a_2,2);
a_2 = [ones(1,size(data,1)) ; a_2];

%2nd layer to 3rd(output) layer activation
z_3 = W_b2*a_2;
% a_3 = sigmoid(z_3); %64x10000(64 pixels)
a_3 = z_3; % For linear decoder
%size(data); %size = 10000x65

% create y matrice
y_mat = data(:,2:end); % ignore the base;10000x64

% Calculate cost
%cost = 1/m*sum(diag(-y_mat*log(a_3)-(1-y_mat)*log(1-(a_3)))); %better error term
cost = 1/(2*m)*sum(sum((y_mat'-a_3).^2)); %<-- this gives a lot of errors

% %######################Calculate regularized cost
% size(W_b1)
% size(W_b2)
regularized_term = lambda/(2)* ...
    (sum(sum(W_b1(:,2:end).^2))+sum(sum(W_b2(:,2:end).^2))); 
cost = cost + regularized_term;

% %##############Calculate the cost for sparsity constraint
sparse_cost = beta*sum(sparsityParam*log(sparsityParam./p_optimal)+ ...
    (1-sparsityParam)*log((1-sparsityParam)./(1-p_optimal)));
cost = cost + sparse_cost;

% -------------------------------------------------------------
% -------------Step two:calculate gradient--------------
% size(data)  %10000x65
% size(W_b1)  %25*65
% size(W_b2)  %64*26
sparse_term = beta*(-sparsityParam./p_optimal+ ...
    (1-sparsityParam)./(1-p_optimal));  % for the sparse constraint

this_y = y_mat'; %size=64x10000
%delta_3 = -(this_y-a_3).*sigmoidGradient(z_3);  %64*10000
delta_3 = -(this_y-a_3);  %for linear decoder

%####Step 3
%size(W_b2(:,2:end)'*delta_3) %size=25x10000
delta_2 = ((W_b2(:,2:end))'*delta_3 + ...
    repmat(sparse_term,1,m)).*sigmoidGradient(z_2); % 25*10000;This is for the case when the square error is used in calculation of cost
%delta_2 = ((W_b2(:,2:end))'*delta_3).*sigmoidGradient(z_t_2); % This
%is for the cost function when KL is used

%######Step 4
tri_2 = delta_3*a_2'; %64*26
tri_1 = delta_2*data;  %25*65

%#####Step5;
% DO not regularized the bias term
Theta1_grad = 1/m*tri_1 + lambda*W_b1; %size=25*65
Theta2_grad = 1/m*tri_2 + lambda*W_b2; %size=64*26
% no regularization on the bias term
Theta1_grad(:,1) = Theta1_grad(:,1) - lambda*W_b1(:,1);
Theta2_grad(:,1) = Theta2_grad(:,1) - lambda*W_b2(:,1);
%Assign value
W1grad = Theta1_grad(:,2:end);
W2grad = Theta2_grad(:,2:end);
b1grad = Theta1_grad(:,1);
b2grad = Theta2_grad(:,1);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

% %-------------------------------------------------------------------
% % Here's an implementation of the sigmoid function, which you may find useful
% % in your computation of the costs and the gradients.  This inputs a (row or
% % column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).
%
% function sigm = sigmoid(x)
%
%     sigm = 1 ./ (1 + exp(-x));
% end
                                  

