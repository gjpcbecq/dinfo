function [hXkY, hX, miXY] = hc(x, y, varargin)
% Compute conditional entropy h(x | y) 
% 
% Syntax
%
% [hXkY, hX, miXY] = hc(x, y, k=1)
%
% Input
%
% x: nDimX, nObs
% y: nDimY, nObs
% varargin: 
%  1 - k=1: int, number of neighbors
%
% Output
% 
% hXkY: float
% hX: float
% miXY: float
% 
% Description
%
% $$ h(x | y) = h(x) - i(x; y) $$
%
% Example
% 
% rng(1)
% [x, y, z] = model.GaussianXYZ(100, 0.9, 0.5, 0.1); 
% hXkY = funKraskov.hc(x, y, 20);
% hYkX = funKraskov.hc(y, x, 20);
% disp(['hXkY: ', num2str(hXkY)]);
% disp(['hYkX: ', num2str(hYkX)]);
% 
%   hXkY: 0.71356
%   hYkX: 0.71677
%
if (nargin == 2), 
    k = 10; 
end
if (nargin == 3), 
    k = varargin{1}; 
end
%___________
% 1st method
% xy = [x; y]; 
% hXY = funKraskov.h(xy, k);
% hY = funKraskov.h(y, k);
% hXkY = hXY - hY;
% return [hXkY, hX, hXY]
%___________
%___________
% 2nd method
hX = funKraskov.h(x, k);
miXY = funKraskov.mi(x, y, k);
hXkY = hX - miXY; 
%___________

return 