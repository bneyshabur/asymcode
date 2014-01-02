function s = signnz(x)
% Returns a +-1 vector of signs of x, setting zeros to random signs
% 
% Input:
% x:  A real-valued vector
%
% s:  A +-1 vector of signs of x, setting zeros to random signs    
  
  s=sign(x);
  zerosins = (s==0);
  if any(zerosins),
    s(zerosins)= 2*(rand(1,nnz(zerosins))>0.5)-1;
  end

end 
