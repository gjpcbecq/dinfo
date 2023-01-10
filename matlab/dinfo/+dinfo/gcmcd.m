function dXYKZ = gcmcd(x, y, z, p, varargin)
% Compute Geweke's dynamic causal measure conditionally to z, Gx>y|z 
% 
% Syntax
% 
% dXYKZ = gcmcd(x, y, z, p, kernelMethod='Gaussian', nFold=10, 
%   listLambda=1, param={1})
% 
% Input 
% 
% x: nDimX-by-nObs
% y: 1-by-nObs
% z: nDimZ-by-nObs
% p: integer, order of the model 
% varargin: 
%   1 kernelMethod='Gaussian': 'Gaussian', 'linear'
%   2 nFold=10: integer, number of parts in cross-validation
%   3 listLambda=1: 1-by-nLambda, array of optimization parameter
%   4 param={1}: 1-by-nParam, cell array of arrays of kernelMethod parameters
% 
% Output
% 
% dXYKZ: float
% 
% Description
%
% $$ G_{X \rightarrow Y || Z} = \frac{\sigma^2(y_t | y^{t - 1}, z^{t - 1})}
%   {\sigma^2(y_t | y^{t - 1}, x^{t - 1}, z^{t - 1})}$$ 
% Wrapper to funGewekeKernel.gcmcd 
%
% Example 
% 
% x = [2, 3, 4, 5, 6, 7, 8];
% y = [1, 2, 3, 4, 5, 6, 7]; 
% z = [11, 12, 13, 14, 15, 16, 17; 21, 22, 23, 24, 25, 26, 27];
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% kernelMethod = 'Gaussian'; 
% dXYKZ = dinfo.gcmcd(x, y, z, 1, kernelMethod, 2, listLambda, param);
% dYXKZ = dinfo.gcmcd(y, x, z, 1, kernelMethod, 2, listLambda, param);
% disp(['dXYKZ: ', num2str(dXYKZ), ' dYXKZ: ', num2str(dYXKZ)]); 
% kernelMethod = 'linear'; 
% dXYKZ = dinfo.gcmcd(x, y, z, 1, kernelMethod, 2, listLambda, param);
% dYXKZ = dinfo.gcmcd(y, x, z, 1, kernelMethod, 2, listLambda, param);
% disp(['dXYKZ: ', num2str(dXYKZ), ' dYXKZ: ', num2str(dYXKZ)]); 
%
%   dXYKZ: -0.31868 dYXKZ: -0.32304
%   dXYKZ: 1.1006 dYXKZ: 1.33
% 
% Example
% 
% rng(1)
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.gcmcd(x, y, z, 1, 'Gaussian', 2, listLambda, param));
%
%   0.0349
%
% See also funGewekeKernel.gcmcd
%
if (nargin == 4), 
    kernelMethod = 'Gaussian'; 
    nFold = 10; 
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 5),
    kernelMethod = varargin{1};
    nFold = 10; 
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 6), 
    kernelMethod = varargin{1};
    nFold = varargin{2}; 
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 7), 
    kernelMethod = varargin{1};
    nFold = varargin{2}; 
    listLambda = varargin{3}; 
    param = {1}; 
end
if (nargin == 8), 
    kernelMethod = varargin{1};
    nFold = varargin{2}; 
    listLambda = varargin{3}; 
    param = varargin{4}; 
end
if (nargin > 8), 
    disp('warning... check parameters'); 
    dXYKZ = nan; 
    return
end

dXYKZ = funGewekeKernel.gcmcd(x, y, z, p, kernelMethod, nFold, listLambda, param); 
return
