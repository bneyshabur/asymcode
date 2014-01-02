function [wq, wd] = asymmaxcutW( S, X, Wq, Wd, loss, param, initwq, initwd )
% This function finds linear separators to generate a new bit for query and database objects.
%
% Input:
% S:            An N*N similarity matrix of objects in the training set
% X:            A d*N matrix of features for objects in the training set
% Wq:           The current predictor for query objects
% Wd:           The current predictor for database objects
% loss:         The handle to the loss function. loss(S,Y) returns the loss of prediting the score matrix Y for the similarity matrix S.
% param:        An structure that includes the input parameters of the method
% initwq:       The initial value for the linear separator of new bits of query objects
% initwq:       The initial value for the linear separator of new bits of database objects
%
% Output:
% initwq:       The linear separator of the new bit for each query object
% initwq:       The linear separator of the new bit for each database object
  
  blocksize = param.blocksize;  % The size of random block for updates
  N = size(X,2);                % The total number of objects in the training set
  wq = initwq;                  % Linear separator for one bit in the query
  wd = initwd;                  % Linear separator for one bit in the database
  iter = 1;                     % Iteration number
  max_objval = -inf;            % Maximum objective value we've achieved 
  max_iter = 0;                 % The iteration number that we achieved the maximum objective value

  % at each loop, we update both wq and wd
  while(1)
    
    % Chooses a random subset of objects
    obj1 = randperm(N,blocksize);
    obj2 = randperm(N,blocksize);

    % Generates the code for selected objects
    U = signnz( Wq * X(:,obj1) );
    V = signnz( Wd * X(:,obj2) );

    % Generates the prediciton matrix
    Yp = U' * V;

    % Transfroms a 0-1 similarit matrix into a sign similarity matrix
    Sp = 2 * S(obj1,obj2) - 1;

    % Calculates the difference between predicting 1 vs predicting -1 for each entry. This corresponds to calculating the gradient
    M = loss(Sp,Yp-1) - loss(Sp,Yp+1);
  
    % Updates wq
    % -----------

    % Generates bits for database objects
    r = signnz( X(:,obj2)' * wd );

    Mr = M * r;
    
    % The desired bit for each query object
    y = signnz(Mr);

    % The weight of each query objects. This corresponds to the contribution of each bit to the objective function
    weights = abs(Mr);
    
    % Learning wq by solivng a weighted linear classifiction problem using hinge loss
    wq = hingeW( y, X(:,obj1), weights, wq, param );
    
    % -----------
  
    %Updates wd
    % -----------

    % Generates bits for database objects
    l = signnz( X(:,obj1)' * wq );

    lM = M' * l;
    
    % The desired bit for each database object
    y = signnz(lM);

    % The weight of each database objects. This corresponds to the contribution of each bit to the objective function
    weights = abs(lM);
    
    % Learning wd by solivng a weighted linear classifiction problem using hinge loss
    wd = hingeW( y, X(:,obj2), weights, wd, param);
    
    % -----------

    % Updates bits for query objects
    r = signnz( X(:,obj2)' * wd );

    % Calculates the objective value for current bits (here we want to maximize this objective)
    objval = lM' * r;
    
    % Checks whether or not we are still improving the objective value
    if( objval > max_objval )
      max_objval = objval;
      max_iter = iter;
    end 
    
    % break if we are not improving the objective in the last few steps
    if  iter - max_iter > 5
      %keyboard
      break;
    end
    
    iter = iter + 1;

  end % while

end

% ================================================
% ================================================
function w = hingeW( y, x, weights, winit, param )
% This function uses hinge loss and gradient descent updates to find a linear separator in a weighted classification problem
%
% Input:
% y:          The array of labels for N objects
% x:          A d*N matrix where each column is the array of features for each object
% weights:    The weight of each object in the objective value. The loss of wrong prediction for each object is equal to the weight of each object
% winit:      The initial value for the linear separator
% param:      It includes the set of input parameteres of the method
%
% Output:
% w:          The linear separator for the weighted classification problem

  N = size(y,1);                  % The number of data points 
  batchsize = param.batchsize;    % Batch size in SGD
  nbatches = param.nbatches;      % Number of batches in SGD
  nepoch = param.nepoch;          % Nubmer of epoches in SGD
  w = winit;                      % the linear separator
  eta = 2;                        % Step-size in SGD
  
  % SGD updates for each epoch
  for t = 1:nepoch
    
    % Updates the step-size
    eta = eta * ( 1-(t-1) / nepoch );
    
    % SGD updates for each batch
    for i = 1:nbatches
      
      % Seletcs the batch
      batch= ceil( N * rand(batchsize,1) );
      
      % Finds the violated predictions
      violated = ( (x(:,batch)' * w) .* y(batch) < 1 );
      
      % Calculates the gradient based on the wrong predictions
      gradient = -x(:,batch) * ( weights(batch) .* y(batch) .* violated );
      
      % Update the linear separator
      w = w - eta * gradient/batchsize;
    
    end % for i

  end % for t

end
