function k = vectorGaussianKernel(wi, w, beta)
% Compute the vector of kernel function k(wi, w(:, j))
% 
% Syntax
%
% k = vectorGaussianKernel(wi, w, beta)
%
% Input
%
% wi: nDim-by-1
% w: nDim-by-nObs
% beta: kernel parameter
% 
% Output
%
% k: 1-by-nObs
% 
% Example
% 
% x = [1, 2, 3; 11, 12, 13]; 
% beta = 2; 
% G = kernel.GaussianGram(x, beta); 
% x1 = kernel.vectorGaussianKernel(x(:, 1), x, beta); 
% x2 = kernel.vectorGaussianKernel(x(:, 2), x, beta); 
% x3 = kernel.vectorGaussianKernel(x(:, 3), x, beta); 
% disp(G)
% disp(x1)
% disp(x2)
% disp(x3)
%
% 1.0000    0.6065    0.1353
% 0.6065    1.0000    0.6065
% 0.1353    0.6065    1.0000
% 
% 1.0000    0.6065    0.1353
% 
% 0.6065    1.0000    0.6065
% 
% 0.1353    0.6065    1.0000
%
% See also kernel.GaussianGram, kernel.kernelGaussian
% 
nObs = size(w, 2); 
k = zeros(1, nObs); 
for j = 1 : nObs, 
    k(j) = kernel.kernelGaussian(wi, w(:, j), beta); 
end
return 
