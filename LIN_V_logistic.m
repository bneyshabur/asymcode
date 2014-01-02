function [Wq, V] = LINLIN_logsitic( X, S, K, param, verbose )
% Given a binary similarity matrix S, the bit length K and feature vectors X, this function returns Wq to be used as linear hash functions for query and V as the binary code for database objects such that the hamming distance of binary codes between query and databased objects corresponds to the similarity matrix.
% 
% Input:
% S:          The similarity matrix which is an N*N sign matrix
% K:          Number of bits in the factorization
% X:          A d*N matrix of features where each column is the set of features for an object
% param:      It includes parameters of the method
% verbose:    If verbose=1 then the program outputs precision and recall at each step
%
% Output:
% Wq:         A K*d matrix we use to generate the binary codes for queries where sign(Wq*X) will be the binary codes for all objects in the training set as queries.
% V:          A K*N matrix that is the binary codes for all objects in the training set as database objects.

  % Adding the path to utilities folder
  addpath utils;          

  % Setting the values for different parameters and variables
  beta = param.beta;                      % The weight on the positive pairs. 1-beta is the weight on negative pairs
  conv = param.conv;                      % The convexity of the loss function. If conv==1 the loss function is convex but for smaller values of conv the loss function is tighter
  epsilon = param.epsilon;                % The precision under which we calculate loss
  windowsize = param.windowsize;          % The number of updates over all bits and ubjects without improvement of loss as the exit condition
  nsamples = param.nsamples;              % The nubmer of random samples we use to update parameters or find some statistics at each step
     
  counter = 0;                            % Counts the number of times that we've updated all bits for all ubjects
  min_loss  = 1;                          % This variable stores the minimum loss over all iterations
  curr_loss = 1;                          % This variable stroes the value of loss at the current iteration
  min_iter = 0;                           % min_iter stores the iteration number where we had the minimum loss

  [d,N] = size(X);                        % d is the number of features and N is the number of objects
  X = [X;ones(1,N)];                      % Adding one column of 1s to X that is correspond to learning biases for features
  d = d + 1;                              % Updating d

  Wq = [ randn(K,d-1) zeros(K,1)];        % Initializes Wq to be random
  Wq = Wq / norm(Wq);                     % Normalizes Wq
  V = signnz(Wq * X);                     % Initializes V to be equal to the query binary codes (symmetric initialization)
  
  D.BP = zeros(K+1,1);                    % BP(i) is the number of entries in Y that are equal to 2*i-K+2 and their label is positive where Y is the matrix of inner product of binary codes
  D.BN = zeros(K+1,1);                    % BN(i) is the number of entries in Y that are equal to 2*i-K+2 and their label is negative where Y is the matrix of inner product of binary codes
 
  weighting = S.*(beta - 0.5) + 0.5;      % Setting the weights such that betta is the weight of positive pairs and (1-betta) is the weight of negative pairs.


  % Returns the handle to the loss function for the given prediction matrix, similarity matrix and the threshold
  subloss = @(theta,sp,yp) ( (sp.*(beta-0.5)+0.5) .* ( log(1+exp(-sp .* (yp-theta) ) ).^conv ) );

  % Calculates the needed statistics of the prediciton using Wd and Wq parameters
  D = updateD(X,S,Wq,V,K,nsamples,subloss);
  
  % Calculates the best threshold for the current predictions  
  [precision,recall,theta] = findtheta(D,subloss);

  % The handle to the loss function using the current threshold
  loss = @(Sp,Yp)( subloss(theta,Sp,Yp) );
 

  while(1)

    counter = counter + 1;

    fprintf(1,'Iteration: %d\n',counter);

  
    % Calculates a random permutation that is the order we update the bits
    perm =  randperm(K); 

    % At each loop i, we update the bit perm(i) for all objects
    for i=1:K
      tic;

      % Selects a bit from the permutation
      ind = perm(i);

      % Find new values for one row of Wq and Wd when other rows are fixed.
      [wq, v] = updateWV( S, X, Wq(1:end~=ind,:), V(1:end~=ind,:), loss, param, Wq(ind,:)', V(ind,:)' );

      % Calculating and storing the new setting
      %----------------------------------       
      
      % Storing the new parameters
      new_Wq=Wq;
      new_V=V;
      new_Wq(ind,:)=wq';
      new_V(ind,:)=v';
      
      % Calculating needed statistics using new parameters
      new_D = updateD( X, S, new_Wq, V,K, nsamples, subloss);
      
      % Calculates the best threshold for the new predictions  
      [new_recall,new_precision,new_theta] = findtheta(new_D,subloss);

      % The handle to the loss function using the new threshold
      new_loss = @(Sp,Yp)( subloss(new_theta,Sp,Yp) );

      % Calculating the total loss for the training data using statistics in newD
      next_loss = sumloss(new_D,new_loss);
      
      %---------------------------------- 
      
      % Updates parameters if the new loss is not much worse than previous one
      if( next_loss < curr_loss + 0.001 )
        Wq = new_Wq;
        V = new_V;
        D = new_D;
        theta = new_theta;
        curr_loss = next_loss;
        precision = new_precision;
        recall = new_recall;
      end
 
      % Calculates the precision, recall and average precision and reports them
      if verbose
        [p_dd,r_dd,avep] = aveprecision(D);
        fprintf(1,'%3d- theta=%3d, loss=%5.4f, precision=%5.4f, recall=%5.4f, Average Precisiton=%5.4f : %3.0f sec\n', i, theta, curr_loss, precision, recall ,avep , toc);
      end
      
      % Updates min_loss if we have achevied a signifiantly better loss
      if( curr_loss <= min_loss - epsilon )
        min_loss = curr_loss;
        min_iter = counter;
      end

    end % for i

    % The function finishes if we didn't have a significant improvement in the last windowsize+1 iterations
    if( counter - min_iter > windowsize)
      break;
    end
  
  end % while 
end

% =============================================================
% =============================================================
function D = updateD( X1, S, Wq, V, K, nsamples, subloss )
% This function calculates some statistics about the generated binary codes
%
% Input:
% X1:         A d*N1 matrix that is d-dimensional features for N1 objects
% S:          An N1*N2 similarty matrix of two sets of objects
% Wq:         A K*d matrix used in the hash function for query objects
% V:          The binary code for N2 objects as database objects
% K:          The bit length for presentation of objects as query or in database
% nsamples:   The nubmer of random samples we pick to collect statistics
% subloss:    The loss function based on the prediciton, similarity and the threshol
%
% Output:
% D:          A cell strcuture that includes the needed statistics of the current binary codes


  N1 = size(S,1);
  N2 = size(S,2); 

  % Selecting a random subset of samples
  samples = randperm( N1*N2, nsamples );
  s = 2 * S(samples) - 1;
  
  % Generating the associated indices
  [x1ind x2ind] = ind2sub([N1 N2], samples );
  x1 = X1(:,x1ind);
  
  % U is  K*nsamples binary matrix 
  U = sign(Wq*x1);

  % Construct the initial prediction matrix Y 
  y = sum( U .* V(:,x2ind), 1);
 
   % Calculate the necessary values for D 
  for i=1:K+1
    % counting the nubmer of elements in Y with specific values 
    D.BP(i) = size( find( s>0 & y+K+2==2*i), 2 );
    D.BN(i) = size( find( s<0 & y+K+2==2*i), 2 );
  end

  % Total number of positive and negative pairs
  D.NN = sum(sum(s<0));
  D.NP = sum(sum(s>0));

end

% ===========================================================
% ===========================================================
function [pp,rr,theta] = findtheta(D,subloss)
% This function returns the threshold that minimizes the loss
% 
% Input:
% D:        A cell structure that includes the needed statistics about the current codes
% subloss:  A handle to the loss function
% 
% Output:
% theta:    The optimal threshold for the given loss function
% pp:       Precision for threshold theta
% rr:       Recall for threshold theta

  % The number of different values in the prediction matrix Y
  K = max(size(D.BP)) - 1;
  
  % Values of loss for different thresholds
  thloss = zeros(K+2,1);
  
  % Calculating values of loss for different thresholds
  for i=1:K+2
    theta = -K - 3 + 2 * i;
    loss = @(Sp,Yp)( subloss(theta,Sp,Yp) );
    thloss(i) = sumloss(D, loss);
  end

  % Finds the index with minimum loss and the corresponding threshold
  [minval, ind] = min(thloss); 
  theta = -K - 3 + 2 * ind;

  % calculates the precision and recall for the given threshold
  pp = sum( D.BP(ind:end) ) / sum( D.BP(ind:end) + D.BN(ind:end) );
  rr = sum( D.BP(ind:end) ) / D.NP;

end

