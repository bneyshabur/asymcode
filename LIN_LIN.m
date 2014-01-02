function [Wq, Wd] = LINL_LIN( X, S, K )
% Given a binary similarity matrix S, the bit length K and feature vectors X, this function returns Wq and Wd as to be used as linear hash functions for query and database objects such that the hamming distance of binary codes between query and databased objects corresponds to the similarity matrix.
% 
% Input:
% S:              The similarity matrix which is an N*N sign matrix
% K:              Number of bits in the factorization
% X:              A d*N matrix of features where each column is the set of features for an object
%
% Output:
% Wq:             A K*d matrix we use to generate the binary codes for queries where sign(Wq*X) will be the binary codes for all objects in the training set as queries.
% Wd:             A K*d matrix we use to generate the binary codes for database objects where sign(Wd*X) will be the binary codes for all objects in the training set as database objects.


  param.beta = 0.7;             % Weight on the positive pairs. 1-betta is the weight on negative pairs
  param.conv = 0.5;             % The convexity of loss function. If conv==1 the loss function is convex but for smaller values of conv the loss function is tighter
  param.epsilon = 0.001;        % The precision under which we optimize loss
  param.windowsize = 3;         % The number of updates over all bits and ubjects without improvement of loss as the exit condition
  param.blocksize = 1000;       % The size of random block for updates
  param.nsamples = 1000000;     % The number of random samples we use to collect statistics
  param.batchsize = 100;        % Batch size in SGD
  param.nbatches = 100;         % Number of batches in SGD
  param.nepoch = 10;            % Nubmer of epoches in SGD
  verbose = 1;                  % If verbose=1 then the program outputs precision and recall at each step
  % calles the main function with the defined parameteres
  [Wq, Wd] = LIN_LIN_logistic( X, S, K, param, verbose );

end

