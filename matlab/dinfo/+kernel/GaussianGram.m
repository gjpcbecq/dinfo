function G = GaussianGram(A, beta) 
% Compute a Gram matrix with Gaussian kernel from matrix A
% 
% Syntax
% 
% G = GaussianGram(A, beta) 
%
% Input
% 
% A: nDim-by-nObs, A contains the vector in column X(i) = A(:, i). 
% beta: float, parameter of the Gaussian kernel
% 
% Output
% 
% G: the Gram matrix nObs-by-nObs
% 
% Description
%
% The Gram matrix G is given by: 
% $$ G_{i, j} = k(X_i, X_j) $$ with k kernel function and $X_i = A_{., i}$
%
% Example
%
% x = [1., 2., 3.; 11., 12., 13.];
% G = kernel.GaussianGram(x, 2);
% disp(G);
% 
%   1.0000    0.6065    0.1353
%   0.6065    1.0000    0.6065
%   0.1353    0.6065    1.0000
%
N = size(A, 2); 
G = zeros(N); 
for i = 1 : N 
    for j = i : N 
        G(i, j) = kernel.kernelGaussian(A(:, i), A(:, j), beta); 
        G(j, i) = G(i, j); 
    end
end
