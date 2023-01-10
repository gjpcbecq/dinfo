function X = aR1Trivariate(nObs, C, D, S)
% Simulate samples from a trivariate aR1 model 
% 
% Syntax
% 
% X = aR1Trivariate(nObs, C, D, S)
% 
% Input
% 
% nObs: int, number of samples
% C: coupling array
% S: covariance matrix of the noise
% D: constant for the AR
% 
% Output 
% 
% X: (3, nObs)
% 
% Description
% 
% This is the example given in P.-O. Amblard and O. Michel, ``Measuring information flow in networks of stochastic processes'', 2009  
% 
% Example
% 
% rng(1)
% C = [0.4, 0., -0.6; 0.4, 0.5, 0.; 0., 0.5, -0.5]; 
% D = [0, 0, 0]; 
% S = [1, 0.5, 0; 0.5, 1, 0; 0, 0, 1]; 
% X = model.aR1Trivariate(1000, C, D, S);
% disp(corrcoef(X(1, 2:end), X(2, 1:end-1)));
% 
%     1.0000    0.0289
%     0.0289    1.0000
% 
% Example
% 
% rng(1)
% C = [0.4, 0.5, -0.6; 0.4, 0.5, 0.; 0., 0.5, -0.5]; 
% D = [0, 0, 0]; 
% S = [1, 0.5, 0; 0.5, 1, 0; 0, 0, 1]; 
% X = model.aR1Trivariate(1000, C, D, S);
% disp(corrcoef(X(1, 2:end), X(2, 1:end-1)));
% 
%     1.0000    0.6461
%     0.6461    1.0000
% 
M = [0; 0; 0]; 
W = model.GaussianCovariate(nObs, M, S); 
X = zeros(3, nObs); 
for t = 2 : nObs,
    tM1 = t - 1; 
    for i = 1 : 3,
        s = 0;
        for j = 1 : 3, 
            s = s + C(i, j) * X(j, tM1) ; 
        end
        s = s + D(i) + W(i, t);
        X(i, t) = s; 
    end
end
return 
