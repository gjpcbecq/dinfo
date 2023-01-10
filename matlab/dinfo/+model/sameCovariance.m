function C = sameCovariance(nDim, c)
% Generate a covariance matrix with the same correlation for all pairs of variables. 
% 
% Syntax
%
% C = sameCovariance(nDim, c)
% 
% Input
% 
% nDim: int, number of variable
% c: float, value of the correlation between variables 
%
% Output
%
% C: nDim-by-nDim, covariance matrix. 
% 
% Example
%     
% c = 0.9; 
% C = model.sameCovariance(3, c); 
% disp(C)
%
%     1.0000    0.9000    0.9000
%     0.9000    1.0000    0.9000
%     0.9000    0.9000    1.0000
%
C = zeros(nDim); 
for i = 1 : nDim,  
    for j = 1 : nDim,  
        if (i == j),  
            C(i, j) = 1;  
        else
            C(i, j) = c; 
        end
    end
end
return