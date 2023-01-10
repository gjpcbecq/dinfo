function [x, y] = GlassMackey(nObs, epsilonX, epsilonY, alpha)
% Simulate samples from equations based on a Glass Mackey model used in Amblard et al., A Gaussian Process Regression..., 2012 
%
% Syntax
%
% [x, y] = GlassMackey(nObs, epsX, epsY, alpha)
% 
% Input
% 
% nObs: int, number of observations
% epsilonX: 1-by-nObs 
% epsilonY: 1-by-nObs
% alpha: float
% 
% Output
% 
% x: 1-by-nObs
% y: 1-by-nObs
% 
% Description
% 
% The system is given by this equation: 
% $$
% \left \{
% \begin{array}{l}
% x_t = x_{t - 1} - 0.4 \, \left( x_{t - 1} - \frac{2 \, x_{t - 4}} 
%     {1 + x_{t - 4} ^ {10}} \right) \, y_{t - 5} +
%     0.3 \, y_{t - 3} + \epsilon_{x, t} \
% y_t = y_{t - 1} - 0.4 \, \left( y_{t - 1} - \frac{2 \, y_{t - 2}}
%     {1 + y_{t - 2} ^ {10}} \right) + 
%     \alpha \, x_{t - 2} + \epsilon_{y, t}
% \end{array} 
% \right .
% $$
%
% Example
% 
% nObs = 100;
% rng(1)
% epsX = randn(nObs, 1) * 0.01;
% epsY = randn(nObs, 1) * 0.01;
% listLambda = [0.01, 0.1, 1]; 
% param = {[0.1, 1, 10]}; 
% for alpha = [0, 0.01, 0.1, 0.2],
%    [x, y] = model.GlassMackey(nObs, epsX, epsY, alpha);
%    dXY = dinfo.gprcm(x, y, 6, 'Gaussian', listLambda, param); 
%    dYX = dinfo.gprcm(y, x, 6, 'Gaussian', listLambda, param); 
%    disp([alpha, dXY, dYX]);
% end
% 
%        0 -138.1217  -29.5365
%   0.0100 -106.6279  -39.6343
%   0.1000  -76.9470  -40.0221
%   0.2000  -56.5298  -19.7303
%
x = ones(1, nObs + 5); 
y = ones(1, nObs + 5); 
for t = 6 : nObs + 5
    tm1 = t - 1; tm2 = t - 2; tm3 = t - 3; tm4 = t - 4; tm5 = t - 5; 
    x(t) = (x(tm1) - 0.4 * (x(tm1) - (2. * x(tm4)) / ...
        (1. + x(tm4) ^ 10.)) * y(tm5) + 0.3 * y(tm3) + epsilonX(tm5)); 
    y(t) = (y(tm1) - 0.4 * (y(tm1) - (2. * y(tm2)) / ...
        (1. + y(tm2) ^ 10.)) + alpha * x(tm2) + epsilonY(tm5)); 
end
x = x(6:end); 
y = y(6:end); 
return 