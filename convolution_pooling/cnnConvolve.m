function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);    %8
imageDim = size(images, 1);  %64
imageChannels = size(images, 3); %3

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps

    %preprocessing the whitening, W-->WT
    W = W * ZCAWhite;   %400x192;
    %preprocessing the mean substraction for each hidden unit
    units_mean_substrauction = W * meanPatch;   %400x1
    %size(units_mean_substrauction);
% --------------------------------------------------------

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
% 400x8x57x57

for imageNum = 1:numImages %8
  for featureNum = 1:numFeatures %400

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    %57x57
    
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % which is the weight for each hidden node (total 400)
      % ---- YOUR CODE HERE ----
      feature = zeros(8,8); % You should replace this
      
      % Get a particular feature
      this_feature = W(featureNum,:);   %W:400x192
      size(this_feature);   %1x192
      
      % Get feature weight for particular channel
      start_pos = patchDim*patchDim*(channel-1)+1;
      end_pos = patchDim*patchDim*channel;
      feature = reshape(this_feature(start_pos:end_pos),patchDim,patchDim);
      % size(feature)
      % ------------------------

      % Flip the feature matrix because of the definition of convolution, as explained later
      % squeeze:returns an array B with the same elements as A, but with all singleton dimensions removed
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));    %64x64

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
      convolvedImage = convolvedImage + conv2(im,feature,'valid');    %57x57
      % ------------------------
    end
    
    % Subtract(<--add!!!!???) the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----
    convolvedImage = convolvedImage + repmat(b(featureNum)- ...
        units_mean_substrauction(featureNum),size(convolvedImage,1),...
        size(convolvedImage,2));
    
    %apply sigmoid function
    convolvedImage = sigmoid(convolvedImage);
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
end


end

