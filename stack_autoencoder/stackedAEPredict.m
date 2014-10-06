function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
% size(stack{1}.w)    %20x784
% size(data)  %784x10000
% size(stack{1}.b)    %20x1
n = numel(stack);

z=cell(1,n+1);
a=cell(1,n+1);
a{1} = data;
for k = 1:n
   num_col = size(a{k},2);
   z{k+1} = stack{k}.w*a{k} + repmat(stack{k}.b,1,num_col);
   a{k+1} = sigmoid(z{k+1});
end

prob = softmaxTheta * a{n+1}; 
[values,pos] = max(prob,[],1);
pred = pos;

% -----------------------------------------------------------

end


% % You might find this useful
% function sigm = sigmoid(x)
%     sigm = 1 ./ (1 + exp(-x));
% end
