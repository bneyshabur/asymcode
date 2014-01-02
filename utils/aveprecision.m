function [p,r,avep] = aveprecision(D)
% This function calculate the average precision that is the area under the precision-recall curve if the connect the points for consecutive threshold to each other
%
% Input:
% D:        A structure that stors the current setting
% 
% Output:
% p:        The array of precision values for different thresholds
% r:        The array of recall values of different thresholds
% avep:     The average precison ( area under precison-recall curve)

  % number of different values in Y
  num = max(size(D.BP));
  NP = sum(D.BP);

  % number of elements in Y with value greater than threshold
  pos =0;

  % True-positives
  TP = 0;

  p = zeros(num+1,1);
  r = zeros(num+1,1);
  p(num+1) = 1;

  % calculate precision and recall for each possible threshold
  for i=num:-1:1
    pos = pos + ( D.BP(i) + D.BN(i) );
    TP = TP + D.BP(i);
    if( pos > 0)
      p(i) = TP / pos;
    else
      p(i) = 1;
    end
    r(i) = TP / NP;
  end

  % Trapezoidal numerical integration
  avep = abs(trapz(r,p));

end
