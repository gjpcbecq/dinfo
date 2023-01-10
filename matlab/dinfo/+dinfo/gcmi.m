function dXY = gcmi(x, y, p, varargin)
% Compute Geweke's instantaneous causal measure, Gx.y  
% 
% Syntax
% 
% dXY = gcmi(x, y, p, kernelMethod='Gaussian', nFold=10, listLambda=1, 
%   param={1})
% 
% Input 
% 
% x: nDim-by-nObs
% y: 1-by-nObs
% p: int, order of the model 
% varargin: 
%   1 kernelMethod='Gaussian': 'Gaussian', 'linear'
%   2 nFold=10: integer, number of parts in cross-validation
%   3 listLambda=1: 1-by-nLambda, array of optimization parameter
%   4 param={1}: 1-by-nParam, cell array of arrays of kernelMethod parameters
% 
% Output
% 
% dXY: float
% 
% Description
% 
% $$ G_{X . Y} = \frac{\sigma^2(y_t | y^{t - 1}, x^{t - 1})} 
%   {\sigma^2(y_t | y^{t - 1}, x^{t})}$$ 
% wrapper to funGewekeKernel.gcmi
%
% Example 
% 
% x = [2, 3, 4, 5, 6, 7, 8];
% y = [1, 2, 3, 4, 5, 6, 7]; 
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% kernelMethod = 'Gaussian'; 
% dXY = dinfo.gcmi(x, y, 1, kernelMethod, 2, listLambda, param);
% dYX = dinfo.gcmi(y, x, 1, kernelMethod, 2, listLambda, param);
% disp(['dXY: ', num2str(dXY), '; dYX: ', num2str(dYX)]); 
% kernelMethod = 'linear'; 
% dXY = dinfo.gcmi(x, y, 1, kernelMethod, 2, listLambda, param);
% dYX = dinfo.gcmi(y, x, 1, kernelMethod, 2, listLambda, param);
% disp(['dXY: ', num2str(dXY), '; dYX: ', num2str(dYX)]); 
%
%   dXY: -0.43395; dYX: -0.46599
%   dXY: 4.5357; dYX: 0.64893
%
% Example
% 
% rng(1)
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.gcmi(x, y, 1, 'Gaussian', 2, listLambda, param));
%
%   1.5586
%
% See also funGewekeKernel.gcmi
% 
if (nargin == 3), 
    kernelMethod = 'Gaussian'; 
    nFold = 10; 
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 4),
    kernelMethod = varargin{1};
    nFold = 10; 
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 5), 
    kernelMethod = varargin{1}; 
    nFold = varargin{2}; 
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 6), 
    kernelMethod = varargin{1}; 
    nFold = varargin{2}; 
    listLambda = varargin{3}; 
    param = {1}; 
end
if (nargin == 7), 
    kernelMethod = varargin{1}; 
    nFold = varargin{2}; 
    listLambda = varargin{3}; 
    param = varargin{4}; 
end
if (nargin > 7), 
    disp('warning... check parameters'); 
    dXY = nan; 
    return
end

dXY = funGewekeKernel.gcmi(x, y, p, kernelMethod, nFold, listLambda, param); 
return