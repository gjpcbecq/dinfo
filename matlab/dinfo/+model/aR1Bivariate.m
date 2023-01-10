function X = aR1Bivariate(nObs, cXX, cXY, cYX, cYY, gVW, sV, sW, dX, dY)
% Simulate samples from a bivariate aR1 model 
% 
% Syntax
% 
% X = aR1Bivariate(nObs, cXX=1, cXY=0, cYX=1, cYY=0, gVW=0, 
%           sV=1, sW=1, dX=0, dY=0)
% 
% Input
% 
% nObs: int, number of samples
% cXX=0: coef from X to X
% cXY=0: coef from Y to X
% cYX=0: coef from X to Y
% cYY=0: coef from Y to Y
% gVW=0: correlation coefficient between noise V and W
% sV=1: standard deviation noise V 
% sW=1: standard deviation noise W
% dX=0: constant on X
% dY=0: constant on Y
% 
% Output 
% 
% X: 2-by-nObs
% 
% Description
% 
% This is the example given in P.-O. Amblard and O. Michel, ``Measuring information flow in networks of stochastic processes'', 2009  
% 
% Example
% 
% rng(1)
% nObs = 1000;
% x = model.aR1Bivariate(nObs, 0, 0, 0, 0, 0, 1, 1, 0, 0); 
% disp(corrcoef(x(1, 2:end), x(2, 1:end-1))); 
% 
%     1.0000   -0.0066
%     -0.0066    1.0000
% 
% rng(1)
% nObs = 1000; 
% x = model.aR1Bivariate(nObs, 0, 0.9, 0, 0, 0, 1, 1, 0, 0); 
% disp(corrcoef(x(1, 2:end), x(2, 1:end-1))); 
% 
%     1.0000    0.6734
%     0.6734    1.0000
% 
GammaW = [1, gVW; gVW, 1]; 
M = [0.; 0.]; 
W = model.GaussianCovariate(nObs, M, GammaW); 
iV = 1;
iW = 2;
W(iV, :) = W(iV, :) * sV;
W(iW, :) = W(iW, :) * sW;
x = zeros(1, nObs);
y = zeros(1, nObs);
C = zeros(2, 2);
iX = 1;
iY = 2;
C(iX, iX) = cXX;
C(iX, iY) = cXY;
C(iY, iX) = cYX;
C(iY, iY) = cYY;
for t = 2 : nObs, 
    tM1 = t - 1; 
    x(t) = C(iX, iX) * x(tM1) + C(iX, iY) * y(tM1) + dX + W(iX, t);
    y(t) = C(iY, iX) * x(tM1) + C(iY, iY) * y(tM1) + dY + W(iY, t);
end
X = [x; y];

