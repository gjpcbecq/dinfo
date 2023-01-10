% Some models used in publications and examples 
%
% Reference
% 
% Amblard, P.-O. and Michel, O., ``Measuring information flow in networks of 
% stochastic processes'', arXiv, 2009, 0911.2873
% Amblard, P.-O.; Michel, O. J.; Richard, C. and Honeine, P., ``A Gaussian 
% process regression approach for testing Granger causality between time series 
% data'',  Acoustics, Speech and Signal Processing (ICASSP), 2012 IEEE 
% International Conference on, 3357-3360, 2012.
% Amblard, P.-O.; Vincent, R.; Michel, O. J. and Richard, C., ``Kernelizing 
% Geweke's measures of granger causality'', Machine Learning for Signal 
% Processing (MLSP), 2012 IEEE International Workshop on, 1-6, 2012.
% Amblard, P.-O. and Michel, O., ``Causal conditioning and instantaneous 
% coupling in causality graphs'', Information Sciences, Elsevier, 2014.
% Cover, T. and Thomas, J., ``Elements of information theory'', Wiley Online 
% Library, 1991, 6.
% Frenzel, S. and Pompe, B., ``Partial mutual information for coupling analysis 
% of multivariate time series'', Physical review letters,  APS, 99, 204101, 
% 2007.
% 
% Contain
% 
% GaussianBivariate - generate samples of a Gaussian bivariate model
% GaussianChannel - generate samples of a Gaussian channel
% GaussianCovariate - generate samples of a Gaussian covariate model
% GaussianH - compute the entropy of a Gaussian covariate model
% GaussianXY - generate samples of a Gaussian bivariate model
% GaussianXYZ - generate samples of a Gaussian trivariate model
% GlassMackey - generate samples of a Glass Mackey like system 
% aR1Bivariate - generate samples of bivariate AR model of order 1 
% aR1Trivariate - generate samples of trivariate AR model of order 1
% chain - generate samples of a chained system
% chain2 - generate samples of another chained system
% coupledLorenzSystems - generate samples of a coupled Lorenz system
% fourDimensional - generate samples of a four dimensional system
% sameCovariance - generate a matrix with same covariance 
% 
% Copyright 2014/04/08 G. Becq, Gipsa-lab, UMR 5216, CNRS; P.-O. Amblard, 
% Gipsa-lab, UMR 5216, CNRS; O. Michel, Gipsa-lab, UMR 5216, Grenoble-INP.

% 
% guillaume.becq@gipsa-lab.grenoble-inp.fr
% 
% This software is a computer program whose purpose is to compute directed 
% information and causality measures on multivariates.
% 
% This software is governed by the CeCILL-B license under French law and abiding
% by the rules of distribution of free software. You can use, modify and/ or
% redistribute the software under the terms of the CeCILL-B license as 
% circulated by CEA, CNRS and INRIA at the following URL
% "http://www.cecill.info". 
% 
% As a counterpart to the access to the source code and rights to copy, modify
% and redistribute granted by the license, users are provided only with a
% limited warranty and the software's author, the holder of the economic rights,
% and the successive licensors have only limited liability. 
% 
% In this respect, the user's attention is drawn to the risks associated with
% loading, using, modifying and/or developing or reproducing the software by the
% user in light of its specific status of free software, that may mean that it
% is complicated to manipulate, and that also therefore means that it is 
% reserved for developers and experienced professionals having in-depth computer
% knowledge. Users are therefore encouraged to load and test the software's
% suitability as regards their requirements in conditions enabling the security
% of their systems and/or data to be ensured and, more generally, to use and 
% operate it in the same conditions as regards security.  
% 
% The fact that you are presently reading this means that you have had knowledge
% of the CeCILL-B license and that you accept its terms. 