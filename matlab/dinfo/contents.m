% dinfo
% 
% This software is a set of packages containing functions used to compute 
% measures about directed information and causality. 
% 
% The notation of scope is used to access function in a package using the 
% "." operator. For more help about packages in Matlab see the "User's
% Guide/Object-Oriented programming/Defining and Organizing Classes/Scoping
% Classes with Packages"
%
% For example, the use of the function "dist" in the package "kNN" is done 
% with this syntax: d = kNN.dist(x, 10)
%
% The root directory contains this different packages: 
%
% Contain
%
% dinfo - directed information and causality measure
% model - models and theoretic tools used in articles or in tests
% util - some utility functions for multivariate
% binning - information theory measures using binning
% distance - functions for computing distances
% kNN - k nearest neighbors
% funKraskov - information theory measures using Kraskov's methods
% funFrenzel - information theory measures using Frenzel and Pompe's methods
% kernel - function using kernel methods
% funGewekeKernel - Geweke measures using kernel methods
% gpr - causality measures using Gaussian processes with kernel methods
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