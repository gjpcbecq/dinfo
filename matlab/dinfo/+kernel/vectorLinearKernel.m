function k = vectorLinearKernel(wi, w, varargin)
% Compute the vector of function k(wi, w(.))
% 
% Syntax
%
% k = vectorLinearKernel(wi, w, varargin)
% 
% Input 
% 
% wi: nDim-by-1
% w: nDim-by-nObs
% varargin: 
%  param: unusued but necessary for compatibility with other functions 
%   calling it
% 
% Output
% 
% k: 1-by-nObs
%
% Example
%
% wi = [1 ; 11];
% w = [1, 2, 3 ; 11, 12, 13];
% k = kernel.vectorLinearKernel(wi, w - repmat(wi, 1, 3));
% disp(k)
% 
%   0    12    24
% 
% See also kernel.LinearGram, kernel.kernelLinear
%
nObs = size(w, 2); 
k = zeros(1, nObs); 
for j = 1 : nObs 
    k(j) = kernel.kernelLinear(wi, w(:, j)); 
end
return