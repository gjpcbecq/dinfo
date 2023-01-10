function s = hN(N) 
% Compute the opposite of the sum of inverse from 1 to N
% 
% Syntax
% 
% s = hN(N) 
% 
% Input
% 
% N: int
% 
% Output
% 
% s: float
% 
% Description
% 
% $$ h_N = - \sum_{n=1}^{N} n^{-1} $$
% See Frenzel and Pompe's article for details. 
% 
% Example
%
% disp(funFrenzel.hN(100))
%
%   -5.1874
% 
s = 0; 
if (N > 0)
    for i = 1 : N, 
        s = s + 1 / i; 
    end
    s = -s; 
end
return  