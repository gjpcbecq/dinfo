function [micXYKZ, nXZ, nYZ, nZ] = mic(x, y, z, k, metric)
% Compute conditional mutual information or partial mutual information
%
% Syntax
%
% [micXYKZ, nXZ, nYZ, nZ] = mic(x, y, z, k=1, metric='Euclidean')
%
% Input
%
% x: nDimX-by-nObs
% y: nDimY-by-nObs
% z: nDimZ-by-nObs
% k: number of neighbors
% metric: 'Euclidean' or "max'
% 
% Output
%
% miXYKZ: float 
% nXZ: int
% nYZ: int
% nZ: int
%
% Description
% 
% I(X, Y |Z) in the article. 
% $$ \hat{I}(X, Y|Z) = \langle h_{N_{xz}(t)} + h_{N_{yz}(t)} - 
%     h_{N_{z}(t)} \rangle - h_{k - 1} $$
% See Frenzel and Pompe's article for details. 
%
% Example
%
% x = [1, 2, 3; 11, 12, 13; 21, 22, 23];
% y = [3, 4, 9; 13, 14, 19; 23, 24, 29];
% z = [6, 7, 8; 16, 17, 18; 26, 27, 28];
% [micXYKZ, nXZ, nYZ, nZ] = funFrenzel.mic(x, y, z, 2, 'Euclidean'); 
% disp([micXYKZ, nXZ, nYZ, nZ]); 
% 
%     0     3     2     3
%
% Example
% 
% rng(1)
% x = rand(10, 1000);
% y = rand(10, 1000);
% z = rand(3, 1000);
% [micXYKZ, nXZ, nYZ, nZ] = funFrenzel.mic(x, y, z, 10, 'Euclidean');
% disp([micXYKZ, nXZ, nYZ, nZ]); 
%
%    0.0117   53.0000  122.0000  897.0000
% 
% Example
%
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1);
% [micXYKZ, nXZ, nYZ, nZ] = funFrenzel.mic(x, y, z, 10, 'Euclidean');
% disp([micXYKZ, nXZ, nYZ, nZ]); 
% 
%   0.8160   11.0000   12.0000   25.0000
%
hN = @funFrenzel.hN; 
nObs = size(x, 2); 
switch metric, 
    case 'Euclidean', 
        funDist = @distance.Euclid_xXI; 
    case 'max', 
        funDist = @distance.max_xXI; 
end
sumHN = 0; 
dXZ = zeros(nObs, 1); 
dYZ = zeros(nObs, 1); 
dXYZ = zeros(nObs, 1); 
for iObs = 1 : nObs,  
    dX = funDist(x, x(:, iObs)); 
    dY = funDist(y, y(:, iObs)); 
    dZ = funDist(z, z(:, iObs)); 
    for jObs = 1 : nObs,  
        dXYZ(jObs) = max([dX(jObs), dY(jObs), dZ(jObs)]);
        dXZ(jObs) = max([dX(jObs), dZ(jObs)]);
        dYZ(jObs) = max([dY(jObs), dZ(jObs)]);
    end
    [~, iXYZSorted] = sort(dXYZ);
    iXYZKNN = iXYZSorted(k + 1);
    epsilonK = dXYZ(iXYZKNN); 
    nXZ = 0;
    nYZ = 0;
    nZ = 0;
    for i = 1 : nObs,  
        if (dXZ(i) < epsilonK),
            nXZ = nXZ + 1; 
        end
        if (dYZ(i) < epsilonK),
            nYZ = nYZ + 1; 
        end
        if (dZ(i) < epsilonK),
            nZ = nZ + 1; 
        end
    end
    % remove 1 from null distance to himself. 
    sumHN = sumHN + hN(nXZ - 1) + hN(nYZ - 1) - hN(nZ - 1); 
    meanHN = sumHN / nObs; 
end
micXYKZ = meanHN - hN(k - 1); 
return 