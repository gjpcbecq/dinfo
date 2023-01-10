function mmspe = mspe_crossfold(x, w, kernelMethod, nFold, lambda, param)
% Compute the mean squared prediciton error with kernel methods using cross validation
% 
% Syntax
% 
% mmspe = mspe_crossfold(x, w, kernelMethod, nFold, lambda, param)
%
% Input
% 
% x: 1-by-nObs, 1 dimension, targets 
% w: nDimW-by-nObs, predictors
% kernelMethod: 'Gaussian' or 'linear'
% nFold: int, number of fold the cross validation is applied. 
% lambda: float, optimization parameter
% param: 1-by-nParam, kernel parameters
% 
% Output
% 
% mmspe: float, mean of the mean squared error over the nFold 
%   cross-validation
% 
% Description
% 
% x is a set of target and w a set of predictors. 
% x is cut into nFold parts such that : 
% [x[0] ... part[0] ...][... part[i] ... ][... part[nFold-1] ... x[n-1]]
% The same is done for w. 
% Learning is done on the nFold-1 parts and tested on the last one. 
% The learning and test sets are exchanged nFold times.  
% The signal must be stationary on each part to ensure appropriate learning and good performances. % 
%
% Example
% 
% rng(1)
% x = [1:100]; 
% w = [0:99 ;  ones(1, 100)]; 
% iPerm = randperm(100); % need a stationary signal
% x = x(:, iPerm);
% w = w(:, iPerm);
% listLambda = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
% listBeta = [1e1, 1e2, 1e3, 1e4, 1e5];
% mmspe = zeros(6, 5); 
% i = 0; 
% for lambda = listLambda
%    i = i + 1; 
%    j = 0; 
%    for beta = listBeta
%       j = j + 1; 
%       mmspe(i, j) = kernel.mspe_crossfold(x, w, 'Gaussian', 10, ...
%           lambda, beta); 
%    end
% end
% disp(log(mmspe)); 
%
%     6.2175    2.7536    6.5219    6.8033    6.8064
%     5.9727    1.4513    4.9783    6.7768    6.8071
%     5.8499    0.0563    1.2973    6.5226    6.8044
%     5.7874   -2.2367   -3.1090    4.9753    6.7770
%     5.7751   -3.4853   -6.4075    1.2886    6.5226
%     5.7736   -4.1147   -7.2491   -3.1945    4.9752
% 
% See also kernel.mspe, kernel.mspe_crossfold_search
%
if (nFold < 2)
    disp('nFold < 2'); 
    mmspe = nan; 
    return
end
nObs = size(x, 2); 
if (nFold > nObs)
    disp('nFold > nObs')
    mmspe = nan; 
    return
end
% disp([nObs, nFold])
[iL, nSet] = xFoldGetIL(nObs, nFold); 
A = xFoldGetA(nSet, nFold); 
mmspe = 0; 
for i = 1 : nFold
    L = xFoldGetL(iL(i), nSet); 
    T = xFoldGetT(L, A); 
    % disp([A(1), L(1), T(1)])
    xL = x(:, L);
    wL = w(:, L);
    xT = x(:, T);
    wT = w(:, T);
    [mmspeT, ~] = kernel.mspe(xL, wL, xT, wT, kernelMethod, lambda, param); 
    mmspe = mmspe + mmspeT; 
end
mmspe = mmspe / nFold; 
return 

function [iL, nSet] = xFoldGetIL(nObs, nFold)
nSet = fix(nObs / nFold);
iL = zeros(nFold);
k = 1;
for i = 1 : nFold
    iL(i) = k; 
    k = k + nSet;
end
return 

function A = xFoldGetA(nSet, nFold)
A = 1 : nSet * nFold;
return 

function L = xFoldGetL(iL, nSet)
L = iL : iL + nSet -1;
return 

function T = xFoldGetT(L, A)
T = setdiff(A, L);
return 
