function k = kernelGaussian(x, y, beta)
% Compute a Gaussian kernel between vector x and y
%
% Syntax
% 
% k = kernelGaussian(x, y, beta)
% 
% Input
% 
% x: nDim-by-1
% y: nDim-by-1
% beta : scalar,
% 
% Output
%
% k: float
% 
% Description
% 
% $$ k(x,y) = \exp{(- ||x - y||^2 / beta ^ 2)} $$
% 
% Example
% 
% x = [1, 2, 3]';
% y = x; 
% disp(kernel.kernelGaussian(x, y, 1)); 
% 
%   1
% 
% Example
% 
% x = [1, 2, 3]';
% y = [2, 3, 4]'; 
% disp(kernel.kernelGaussian(x, y, 1)); 
%
%   0.0498
%
% Example
% 
% x = [1, 2, 3]';
% y = [2, 3, 4]'; 
% disp(kernel.kernelGaussian(x, y, 3)); 
%
%   0.7165
%
nDim = size(x, 1); 
norm = 0.0; 
for i = 1 : nDim 
    norm = norm + (x(i) - y(i)) ^ 2.0; 
end
k = exp(- norm / (beta ^ 2.0)); 
return 