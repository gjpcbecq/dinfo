function [mspeMin, lambdaMin, paramMin] = mspe_crossfold_search(...
    x, w, kernelMethod, nFold, listLambda, param)
% Compute the mean squared prediciton error for Gaussian kernel with cross-validation and search of the minimum for parameters in the given lists.
% 
% Syntax
%
% [mspeMin, lambdaMin, paramMin] = mspe_crossfold_search(...
%     x, w, kernelMethod, nFold, listLambda, param)
%
% Input
% 
% x: 1-by-nObs, targets
% w: nDimW-by-nObs, predictors
% kernelMethod: 'Gaussian' or 'linear'
% nFold: int, number of sets for crossfold validation 
% listLambda: 1-by-nLambda, list of float for optimization parameter
% param: 1-by-nParam, cell array of arrays of kernel parameter
% 
% Output
% 
% mspeMin: float, the minimal mspe
% lambdaMin: float, the optimal optimization parameter
% paramMin: 1-by-nParam, the optimal kernel parameters
% 
% Description
%
% Same as kernel.mspe_crossfold with a search of optimal parameters for 
% lambda an kernel parameters. 
% 
% Example
% 
% rng(1);
% x = rand(1, 100);
% w = rand(2, 100);
% [mspeMin, lambdaMin, betaMin] = kernel.mspe_crossfold_search(x, w, ...
%   'Gaussian', 10, [1., 10.], {[1., 10.]}); 
% disp([mspeMin, lambdaMin, betaMin])
% 
%     0.0972    1.0000   10.0000
% 
% See also kernel.mspe_crossfold
%
mspeMin = inf; 
nLambda = size(listLambda, 2); 
listParam = util.getListOfCases(param); 
nParam = size(listParam, 2); 
for iLambda = 1 : nLambda 
    lambda = listLambda(iLambda); 
    for iParam = 1 : nParam, 
        thisParam = listParam(:, iParam); 
        mspe = kernel.mspe_crossfold(x, w, kernelMethod, nFold, ...
            lambda, thisParam); 
        if (mspe < mspeMin)
            paramMin = thisParam; 
            lambdaMin = lambda; 
            mspeMin = mspe; 
        end
    end
end

