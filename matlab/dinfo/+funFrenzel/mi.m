function [miXY, nX, nY] = mi(x, y, k, metric)
% Compute mutual information mi(x; y)
% 
% Syntax 
%
% [miXY, nX, nY] = mi(x, y, k, metric)
%
% Input
% 
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% k: int, number of neighbors
% metric: 'Euclidean' or "max'
%
% Output
% 
% miXY: float 
% nX: int
% nY: int
% 
% Description
% 
% I(X, Y) in the article. 
% $$ \hat{I}(X, Y) = \langle h_{N_{x}(t)} + h_{N_{y}(t)} \rangle - 
%   h_{T-1} - h_{k - 1} $$
% See Frenzel and Pompe's article for details. 
%
% Example
%
% x = [1, 2, 3; 11, 12, 13; 21, 22, 23];
% y = [3, 4, 9; 13, 14, 19; 23, 24, 29];
% [miXY, nX, nY] = funFrenzel.mi(x, y, 2, 'Euclidean'); 
% disp([miXY, nX, nY]);
% 
%   0     3     2
%
% Example
%
% rng(1)
% x = rand(10, 1000);
% y = rand(10, 1000);
% [miXY, nX, nY] = funFrenzel.mi(x, y, 10, 'Euclidean');
% disp([miXY, nX, nY])
% 
%   0.0047   58.0000  144.0000
%
% Example
%
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1);
% [miXY, nX, nY] = funFrenzel.mi(x, y, 10, 'Euclidean');
% disp([miXY, nX, nY]); 
%
%   0.6940   28.0000   24.0000
% 
hN = @funFrenzel.hN; 
nObs = size(x, 2); 
switch metric 
    case 'Euclidean',
        funDist = @distance.Euclid_xXI; 
    case 'max', 
        funDist = @distance.max_xXI; 
    otherwise, 
        disp('');
        miXY = nan;
        nX = nan;
        nY = nan;
        return 
end
dXY = zeros(nObs, 1);
sumHN = 0;
for iObs = 1 : nObs,  
    dX = funDist(x, x(:, iObs));
    dY = funDist(y, y(:, iObs));
    for jObs = 1 : nObs,  
        dXY(jObs) = max([dX(jObs), dY(jObs)]); 
    end
    [~, iXYSorted] = sort(dXY); 
    iXYKNN = iXYSorted(k + 1); 
    epsilonK = dXY(iXYKNN); 
    nX = 0; nY = 0;
    for i = 1 : nObs,  
        if (dX(i) < epsilonK), 
            nX = nX + 1; 
        end
        if (dY(i) < epsilonK), 
            nY = nY + 1; 
        end
    end
    % remove 1 from null distance to himself. 
    sumHN = sumHN + hN(nX - 1) ;
    sumHN = sumHN + hN(nY - 1) ;
end
meanHN = sumHN / nObs; 
miXY = meanHN - hN(nObs - 1) - hN(k - 1); 
return 
