function hX = h(x, varargin)
% Compute differential entropy, h(X)
% 
% Syntax
%
% hX = h(x, method='bin', param={2})
%
% Input
% 
% x: (nDim, nObs)
% varargin:
%  1 method='bin': {'bin', 'binning'}, {'Kraskov', 'Leonenko'}
%  2 param={2}: cell array, number of bins, number of neighbors...
%      
% Output
% 
% hX: float
%
% Description
%
% $$ h(x) $$
% This is the differential entropy estimator. It is a wrapper to 
% binning.h, funKraskov.h depending on the method used. 
% 
% Example
% 
% rng(1)
% x = 2 * rand(3, 1000);
% hX = dinfo.h(x);
% disp(num2str(hX))
% hX = dinfo.h(x, 'bin', {10});
% disp(num2str(hX))
% hX = dinfo.h(x, 'Kraskov', {10});
% disp(num2str(hX))
% hTh = 3 * log(2); 
% disp(['hTh: ', num2str(hTh)]);  
% 
%   2.0718
%   1.5202
%   2.2942
%   hTh: 2.0794
%
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% disp(dinfo.h(x));
% disp(dinfo.h(x, 'Kraskov', {10})); 
% 
%   1.4511
%   1.3499
%
% See also binning.h, funKraskov.h
% 
if (nargin == 1), 
    method = 'bin'; 
    param = {2}; 
end
if (nargin == 2), 
    method = varargin{1}; 
    param = {2}; 
end
if (nargin == 3), 
    method = varargin{1}; 
    param = varargin{2}; 
end
    
switch method 
    case {'bin', 'binning'},  
        hX = binning.h(x, param{1}); 
    case {'Leonenko', 'Kraskov'}, 
        hX = funKraskov.h(x, param{1}); 
    otherwise 
        disp('unknown method... check arguments'); 
        hX = nan; 
end
return 