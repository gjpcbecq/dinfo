function listOfCases = getListOfCases(cellListParam)
% Get a list of all cases combining lists
% 
% Syntax
%
% listOfCases = getListOfCases(cellListParam)
%
% Input
% 
% cellListParams: cell array of nList of array
% 
% Output
% 
% listOfCases: nList-by-(nL1 * ... * nLnList)
% 
% Example
% 
% param = {[1, 2, 3, 4], [11., 12.], [21, 22, 23]};
% listOfCases = util.getListOfCases(param); 
% disp(listOfCases); 
% 
%   Columns 1 through 14
% 
%      1     2     3     4     1     2     3     4     1     2     3     4     1     2
%     11    11    11    11    12    12    12    12    11    11    11    11    12    12
%     21    21    21    21    21    21    21    21    22    22    22    22    22    22
% 
%   Columns 15 through 24
% 
%      3     4     1     2     3     4     1     2     3     4
%     12    12    11    11    11    11    12    12    12    12
%     22    22    23    23    23    23    23    23    23    23
% 
% """
nList = size(cellListParam, 2);
nParam = zeros(1, nList);
for i = 1 : nList
    nParam(i) = size(cellListParam{i}, 2); 
end
% number of cases 
nCases = prod(nParam); 
listOfCases = zeros(nList, nCases); 
p = 1; 
for i = 1 : nList
    % disp(['i: ', num2str(i)]); 
    for j = 1 : nCases 
        k = floor(mod((j - 1) / p, nParam(i)) + 1); 
        % disp(['k: ', num2str(k)]); 
        listOfCases(i, j) = cellListParam{i}(k); 
    end
    p = p * nParam(i); 
end
return 