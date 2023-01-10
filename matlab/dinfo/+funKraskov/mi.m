function miXY = mi(x, y, varargin) 
% Compute mutual information mi(x; y) 
% 
% Syntax
%
% miXY = mi(x, y, k=1, metric='Euclidean') 
% 
% Input
%
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% k=1: int, number of neighbors
% metric='Euclidean': str, 'Euclidean' or 'max'
% 
% Output
% 
% miXY: float
%
% Description
% 
% wrapper to the funKraskov.mi1
% 
% Example
%
% rng(1)
% x = 3 * rand(3, 100); 
% y = 1/2 * rand(2, 100); 
% miXY = funKraskov.mi(x, y, 10); 
% disp(num2str(miXY)); 
% disp('0 expected'); 
%
%   -1.4211e-14
%   0 expected
%
if (nargin == 2)
    k = 10; 
    metric = 'Euclidean'; 
end
if (nargin == 3)
    k = varargin{1}; 
    metric = 'Euclidean'; 
end
if (nargin == 4)
    k = varargin{1}; 
    metric = varargin{2}; 
end
nObsX = size(x, 2);
nObsY = size(y, 2);
if (nObsX ~= nObsY), 
    miXY = []; 
    return; 
end
miXY = funKraskov.mi1(x, y, k, metric); 
return 

