function d = dist(x, varargin)
% Compute all distances to the k nearest neighbors
% 
% Syntax
%
% d = dist(x, k=1, method='Euclidean')
%
% Input
% 
% x: nDim-by-nObs
% k: number of neighbors
% method: 'Euclidean', 'max'
% 
% Output
% 
% d : k-by-nObs, distances to kNN
%
% Description
% 
% The algorithm relies on the sort function.  
% For each column j of d, the first line corresponds to the distance of x_j 
% to its first neighbor. 
% The algorithm will return zeros if some samples are the same. 
% To avoid that, a weak noise can be added to x. 
% x = x + alpha * numpy.random.rand(x.shape)
% with alpha set to a small value with respect to the range of x. 
% 
% Example 
% 
% rng(1)
% x = rand(1, 100);
% d = kNN.dist(x, 3);
% disp(d(:, 1:10));
%   0.0003    0.0057    0.0028    0.0087    0.0000    0.0060    0.0118    0.0022    0.0009    0.0029
%   0.0022    0.0206    0.0182    0.0111    0.0064    0.0073    0.0164    0.0033    0.0114    0.0057
%   0.0028    0.0259    0.0193    0.0132    0.0075    0.0100    0.0182    0.0300    0.0173    0.0199
%  
if (nargin == 1)
    k = 1;
    method = 'Euclidean';
end
if (nargin == 2)
    k = varargin{1};
    method = 'Euclidean';
end
if (nargin == 3)
    k = varargin{1};
    method = varargin{2};
end

nObs = size(x, 2);
if (k + 1 > nObs)
    kMax = nObs - 1 ; 
    print('kMax set to nObs - 1')
else 
    kMax = k; 
end
d = zeros(kMax, nObs); 
switch method
    case 'Euclidean'
        funDist = @distance.Euclid_xXI; 
    case 'max'
        funDist = @distance.max_xXI; 
end
for i = 1 : nObs
    dist = funDist(x, x(:, i)); 
    sortDist = sort(dist); 
    % 1st value contain distance to itself and is equal to 0
    d(:, i) = sortDist(2 : kMax + 1); 
end
return 