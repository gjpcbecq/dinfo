function s = mse(x, xHat)
% Compute the mean squared error between x and xHat
% 
% Syntax
%
% s = mse(x, xHat)
% 
% Input
%
% x: 1-by-nObs
% xHat: 1-by-nObs
% 
% s: 1-by-1
% 
% Example
% 
% x = [1., 2., 3., 4.];
% xHat = [1.1, 1.9, 3.2, 3.9];
% s = util.mse(x, xHat);
% disp(s); 
%     0.0175
% 
% See also util.msecoef
% 
nObs = size(x, 2); 
s = 0; 
for i = 1 : nObs,  
    s = s + (x(i) - xHat(i)) ^ 2; 
end
s = s / nObs; 
return