function hTh = GaussianH(V)
% Compute Gaussian entropy given V matrix of covariance
% 
% Syntax
% 
% hTh = GaussianH(V)
% 
% Input
% 
% V: int, float, array
% 
% Output
% 
% hTh: float
% 
% Description
%
% $d$ is the dimension of the variable. 
% $V$ is a coefficient of correlation, or the matrix of covariance. 
% $|V|$ is the determinant of V.
% $$ h = \frac{1}{2} \, \log{(2 \, \pi \, e)^d \, |V|} $$
%
% Example 
% 
% V = 1;
% disp(model.GaussianH(V));
%
% 1.4189
% 
% Example
%
% V = [1., 0.8; 0.8, 1]; 
% disp(model.GaussianH(V));
%
% 2.3271
%
nDim = size(V, 1); 
detV = det(V); 
coef = (2 * pi * exp(1)) ^ nDim; 
hTh = 1 / 2 * log(coef * detV); 
return