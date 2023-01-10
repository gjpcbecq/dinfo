function [miXY, epsilon, nX, nY] = mi2(x, y, varargin) 
% Compute mutual information mi(x; y) using 2nd method of Kraskov
%
% Syntax
%
% [miXY, epsilon, nX, nY] = mi2(x, y, k=1, metric='Euclidean') 
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
% epsilon: 3-by-nObs, distance to Z, X and Y 
% nX: 1-by-nObs, number of samples near X
% nY: 1-by-nObs, number of samples near Y
% 
% Description
% 
% See reference for details 
%
% Example
%
% rng(1)
% x = 3 * rand(3, 100); 
% y = 1/2 * rand(2, 100); 
% [miXY, epsilon, nX, nY] = funKraskov.mi2(x, y, 10); 
% disp([num2str(miXY)]); 
% disp('0 expected'); 
%
%   -0.013528
%    0 expected
%
% Example
%
% rng(1);
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% [miXY, epsilon, nX, nY] = funKraskov.mi2(x, y, 10); 
% disp(num2str(miXY)); 
%
%   0.69958
%
if (nargin == 2)
    k = 1; 
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
    miXY = []; epsilon = []; nX = []; nY = []; 
    return; 
end
nObs = nObsX; 
switch metric
    case 'Euclidean'
        funDist = @distance.Euclid_xXI;
    case 'max' 
        funDist = @distance.max_xXI;
end
epsilon = zeros(3, nObs); 
% we remove one point for counting due to the 0 distance from xi to himself. 
nX = zeros(1, nObs) - 1;
nY = zeros(1, nObs) - 1;
IZ = 1; IX = 2; IY = 3; 
for iObs = 1 : nObs,  
    dX = funDist(x, x(:, iObs));
    dY = funDist(y, y(:, iObs));
    dZ = max(dX, dY);
    [~, iZSorted] = sort(dZ); 
    iKNN = iZSorted(k + 1);
    epsilon(IZ, iObs) = dZ(iKNN);
    epsilon(IX, iObs) = max(dX(iZSorted(1:k+1)));
    epsilon(IY, iObs) = max(dY(iZSorted(1:k+1)));
    for i = 1 : nObs 
        if (dX(i) <= epsilon(IX, iObs))
            nX(iObs) = nX(iObs) + 1; 
        end
        if (dY(i) <= epsilon(IY, iObs))
            nY(iObs) = nY(iObs) + 1; 
        end
    end
end
% Eq. 9, Kraskov 2004
psiNxNy = mean(psi(nX) + psi(nY)); 
miXY = psi(k) - 1 / k - psiNxNy + psi(nObs); 
return