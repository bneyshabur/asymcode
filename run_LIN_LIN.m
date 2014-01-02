function run_LIN_LIN
  % This function is just a demo that shows how to call the main function LIN_LIN 

  K = 16;                                 % K is the number of bits
  data = load( 'LabelMe_sample.mat' );    % loading data
  S = sparse(data.S_training);            % S is an N*N similarity matrix
  X = data.X_training;                    % X is a d*N matrix where each column is a set of features for an object

  [Wq,Wd] = LIN_LIN( X, S, K );           % calling the main function
  

  save('parameters','Wq','Wd');           % saving the parameters

end
