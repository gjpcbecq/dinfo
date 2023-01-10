function G = LinearGram(A, varargin)
% Compute a Gram matrix with linear kernel from matrix A
% 
% Syntax
%
% G = LinearGram(A, varargin)
%
% Input
% 
% A: nDim-by-nObs, A contains the vector in column X(i) = A(:, i). 
% varargin: 
%  param: unusued but necessary for compatibility with other functions
%   calling it
% 
% Output
% 
% G: nObs-by-nObs, Gram matrix 
% 
% Description
% 
% The Gram matrix G is given by: 
% $$ G_{i, j} = k(X_i, X_j) $$ with k kernel function and $X_i = A_{., i}$
%
% Example 
% 
% x = [1., 2., 3. ; 11., 12., 13.]; 
% G = kernel.LinearGram(x); 
% disp(G)
% 
%  122   134   146
%  134   148   162
%  146   162   178
% 
% See also kernel.kernelLinear
%
N = size(A, 2);
G = zeros(N);
for i = 1 : N 
    for j = i : N 
        G(i, j) = kernel.kernelLinear(A(:, i), A(:, j)); 
        G(j, i) = G(i, j); 
    end
end
return 