function run_LIN_V
% This function is just a demo that shows how to call the main function LIN_V

  K = 16;                                 % K is the number of bits
  data = load( 'LabelMe_sample.mat' );    % loading data
  S = sparse(data.S_training);            % S is an N*N similarity matrix
  X = data.X_training;                    % X is a d*N matrix where each column is a set of features for an object

  [Wq,V] = LIN_V( X, S, K );              % calling the main function
  

  save('parameters','Wq','V');            % saving the parameters

end
