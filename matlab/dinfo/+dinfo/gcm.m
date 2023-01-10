function dXY = gcm(x, y, z, p, varargin)
% Compute Geweke's causal measure, Gx.y, Gx->y, Gx.y|z or Gx->y|z 
% 
% Syntax
% 
% dinfo.gcm(x, y, z, p, condition='', method='dynamic', 
%   kernelMethod='Gaussian', nFold=10, listLambda=[1.], param={[1.]})
% 
% Input 
% 
% x: nDimX-by-nObs
% y: 1-by-nObs
% z: nDimZ-by-nObs
% p: integer, order of the model 
% varargin: 
%     1 condition='': {'', 'unconditional'}; 'conditional'
%     2 method='dynamic': {'dynamic', 'd'}; {'instantaneous', 'instant', 'd'}
%     3 kernelMethod='Gaussian': 'Gaussian'; 'linear'
%     4 nFold=10: number of parts in cross-validation
%     5 listLambda=[1.]: vector of lambda values
%     6 param={[1.]}: cell array of vector of parameters
% 
% Output
% 
% dXY: float
% 
% Description
%
% wrapper to dinfo.gcmd, dinfo.gcmi, dinfo.gcmcd, dinfo.gcmci
%
% Example 
% 
% x = [2, 3, 4, 5, 6, 7, 8];
% y = [1, 2, 3, 4, 5, 6, 7];
% z = [11, 12, 13, 14, 15, 16, 17; 21, 22, 23, 24, 25, 26, 27];
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% dXY_gcmd = dinfo.gcm(x, y, [], 1, '', 'dynamic', 'Gaussian', ...
%   2, listLambda, param);
% dXY_gcmi = dinfo.gcm(x, y, [], 1, '', 'instant', 'Gaussian', ...
%   2, listLambda, param);
% dXYkZ_gcmcd = dinfo.gcm(x, y, z, 1, 'conditional', 'dynamic', 'Gaussian', ...
%     2, listLambda, param);
% dXYkZ_gcmci = dinfo.gcm(x, y, z, 1, 'conditional', 'instant', 'Gaussian', ...
%     2, listLambda, param);
% disp(['dXY_gcmd: ', num2str(dXY_gcmd), ...
%       ', dXY_gcmi: ', num2str(dXY_gcmi), ...
%       ', dXYkZ_gcmcd: ', num2str(dXYkZ_gcmcd), ...
%       ', dXYkZ_gcmci: ', num2str(dXYkZ_gcmci)]); 
% dXY_gcmd = dinfo.gcm(x, y, [], 1, '', 'dynamic', 'linear', ...
%   2, listLambda, param);
% dXY_gcmi = dinfo.gcm(x, y, [], 1, '', 'instant', 'linear', ...
%   2, listLambda, param);
% dXYkZ_gcmcd = dinfo.gcm(x, y, z, 1, 'conditional', 'dynamic', 'linear', ...
%     2, listLambda, param);
% dXYkZ_gcmci = dinfo.gcm(x, y, z, 1, 'conditional', 'instant', 'linear', ...
%     2, listLambda, param);
% disp(['dXY_gcmd: ', num2str(dXY_gcmd), ...
%       ', dXY_gcmi: ', num2str(dXY_gcmi), ...
%       ', dXYkZ_gcmcd: ', num2str(dXYkZ_gcmcd), ...
%       ', dXYkZ_gcmci: ', num2str(dXYkZ_gcmci)]); 
% 
%   dXY_gcmd: -0.31031, dXY_gcmi: -0.43395, dXYkZ_gcmcd: -0.31868, dXYkZ_gcmci: -0.1429
%   dXY_gcmd: 5.3566, dXY_gcmi: 4.5357, dXYkZ_gcmcd: 1.1006, dXYkZ_gcmci: 0.58424
% 
% Example
%
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1);
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% dXY_gcmd = dinfo.gcm(x, y, [], 1, '', 'dynamic', 'Gaussian', ...
%   2, listLambda, param);
% dXY_gcmi = dinfo.gcm(x, y, [], 1, '', 'instant', 'Gaussian', ...
%   2, listLambda, param);
% dXYkZ_gcmcd = dinfo.gcm(x, y, z, 1, 'conditional', 'dynamic', 'Gaussian', ...
%     2, listLambda, param);
% dXYkZ_gcmci = dinfo.gcm(x, y, z, 1, 'conditional', 'instant', 'Gaussian', ...
%     2, listLambda, param);
% disp(['dXY_gcmd: ', num2str(dXY_gcmd), ...
%       ', dXY_gcmi: ', num2str(dXY_gcmi), ...
%       ', dXYkZ_gcmcd: ', num2str(dXYkZ_gcmcd), ...
%       ', dXYkZ_gcmci: ', num2str(dXYkZ_gcmci)]); 
% 
%   dXY_gcmd: 0.02228, dXY_gcmi: 1.5586, dXYkZ_gcmcd: 0.034858, dXYkZ_gcmci: 3.5733
%
% See also dinfo.gcmd, dinfo.gcmi, dinfo.gcmcd, dinfo.gcmci
% 
if (nargin == 4)
    condition = ''; 
    method = 'dynamic'; 
    kernelMethod = 'Gaussian'; 
    nFold = 10;
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 5)
    condition = varargin{1}; 
    method = 'dynamic'; 
    kernelMethod = 'Gaussian'; 
    nFold = 10;
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 6)
    condition =  varargin{1}; 
    method =  varargin{2}; 
    kernelMethod = 'Gaussian'; 
    nFold = 10;
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 7)
    condition =  varargin{1}; 
    method =  varargin{2}; 
    kernelMethod = varargin{3}; 
    nFold = 10;
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 8)
    condition =  varargin{1}; 
    method =  varargin{2}; 
    kernelMethod = varargin{3}; 
    nFold = varargin{4};
    listLambda = 1; 
    param = {1}; 
end
if (nargin == 9)
    condition =  varargin{1}; 
    method =  varargin{2}; 
    kernelMethod = varargin{3}; 
    nFold = varargin{4};
    listLambda = varargin{5}; 
    param = {1}; 
end
if (nargin == 10)
    condition =  varargin{1}; 
    method =  varargin{2}; 
    kernelMethod = varargin{3}; 
    nFold = varargin{4};
    listLambda = varargin{5}; 
    param = varargin{6}; 
end

switch condition, 
    case {'', 'unconditional'}, 
        switch method 
            case {'dynamic', 'd'}
                dXY = dinfo.gcmd(x, y, p, kernelMethod, nFold, ...
                    listLambda, param); 
            case {'instantaneous', 'instant', 'i'}
                dXY = dinfo.gcmi(x, y, p, kernelMethod, nFold, ...
                    listLambda, param); 
            otherwise
                dXY = nan; 
                disp('unknown method... check arguments'); 
        end
    case {'conditional'}, 
        switch method 
            case {'dynamic', 'd'}
                dXY = dinfo.gcmcd(x, y, z, p, kernelMethod, nFold, ...
                    listLambda, param); 
            case {'instantaneous', 'instant', 'i'}
                dXY = dinfo.gcmci(x, y, z, p, kernelMethod, nFold, ...
                    listLambda, param); 
            otherwise
                dXY = nan; 
                disp('unknown method... check arguments'); 
        end
    otherwise, 
        dXY = nan; 
        disp('unknown condition... check arguments'); 
end
return 