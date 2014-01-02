function sm = sumloss(D,loss);
% Calculate the loss for the given statistics D about the code
%
% Input:
% D:      A cell structure that includes needed statistics about the codes
% loss:   A handle to the loss function
%
% Output:
% sm:     The loss for given D

  % Bit length
  K = size(D.BP,2)-1;
  
  % Possible combinations of similarities and thresholds
  sp = repmat([1 -1],K+1,1);
  
  % Possible predictions for positvie and negative pairs
  yp = [-K:2:K; -K:2:K]';

  % Possible values of loss
  vloss = loss(sp,yp);
  
  % Total loss based on statistics in D
  sm = D.BP * vloss(:,1) + D.BN * vloss(:,2);
  sm = sm/ (D.NN + D.NP);

end
