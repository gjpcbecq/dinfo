function k = kernelLinear(x, y, varargin) 
% Compute a linear kernel between vector x and y
%
% Syntax
%
% k = kernelLinear(x, y, varargin) 
%
% Input
% 
% x: nDim-by-1
% y: nDim-by-1
% varargin: 
%  param: unused but necessary for compatibility with other functions
%   calling it
% 
% Output 
% 
% k: float
% 
% Description
% 
% k(x, y) = x^t * y
% 
% Example
% 
% x = [1; 2; 3];
% y = [2; 3; 4];
% disp(kernel.kernelLinear(x, y)) 
%
%   20
% 
% Example    
% 
% x = [1; 2; 3];
% y = [2; 3; 4];
% disp(kernel.kernelLinear(x, y - x))
% 
%   6
% 
nDim = size(x, 1); 
k = 0.0;
for i = 1 : nDim 
    k = k + x(i) * y(i); 
end
return