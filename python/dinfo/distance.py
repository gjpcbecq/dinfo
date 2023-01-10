"""
Compute distances between points. 

Contain

Euclid_xXI - Euclidean distance between samples in x and one point xI
max_xXI - Max or Tchebychev distance between samples in x and one point xI

Copyright 2014/04/08 G. Becq, Gipsa-lab, UMR 5216, CNRS; P.-O. Amblard, Gipsa-lab, UMR 5216, CNRS; O. Michel, Gipsa-lab, UMR 5216, Grenoble-INP.
"""
"""
guillaume.becq@gipsa-lab.grenoble-inp.fr

This software is a computer program whose purpose is to compute directed information and causality measures on multivariates.

This software is governed by the CeCILL-B license under French law and abiding
by the rules of distribution of free software. You can use, modify and/ or
redistribute the software under the terms of the CeCILL-B license as circulated
by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". 

As a counterpart to the access to the source code and rights to copy, modify and
redistribute granted by the license, users are provided only with a limited
warranty and the software's author, the holder of the economic rights, and the
successive licensors have only limited liability. 

In this respect, the user's attention is drawn to the risks associated with
loading, using, modifying and/or developing or reproducing the software by the
user in light of its specific status of free software, that may mean that it is
complicated to manipulate, and that also therefore means that it is reserved for
developers and experienced professionals having in-depth computer knowledge.
Users are therefore encouraged to load and test the software's suitability as
regards their requirements in conditions enabling the security of their systems
and/or data to be ensured and, more generally, to use and operate it in the same
conditions as regards security.  

The fact that you are presently reading this means that you have had knowledge
of the CeCILL-B license and that you accept its terms.

"""
import numpy 
#_______________________________________________________________________________
def Euclid_xXI(x, xI): 
    """Compute all the distances of samples x to sample xi using Euclidean distance
    
    Syntax
    
    dist = Euclid_xXI(x, xI)
        
    Input
       
    x: (nDim, nObs) an array of observations
    xI: (nDim, 1) or (nDim, ), the observation to evaluate
    
    Output
    
    dist: (nObs, )
    
    Example
    
    >>> x = numpy.array([[1, 2, 3, 4], [11, 12, 13, 14]])
    >>> xI = numpy.array([[5], [15]])
    >>> dist = distance.Euclid_xXI(x, xI)
    >>> print(dist)

        [ 5.65685425  4.24264069  2.82842712  1.41421356]

    """
    nDim = x.shape[0]
    xC = (x - xI.reshape((nDim, 1)))
    dist = numpy.sqrt(numpy.sum(xC ** 2, axis=0))
    return dist
#_______________________________________________________________________________
#_______________________________________________________________________________
def max_xXI(x, xI): 
    """Compute all the distances of samples x to xi using max distance
    
    Syntax
    
    dist = max_xXI(x, xI) 
    
    Input
       
    x: (nDim, nObs) an array of observations
    xI: (nDim, 1) or (nDim, ), the observation to evaluate
    
    Output
    
    dist: float
    
    Example
    
    >>> x = numpy.array([[1, 2, 3, 4], [11, 12, 13, 14]])
    >>> xI = numpy.array([[5], [15]])
    >>> dist = distance.max_xXI(x, xI)
    >>> print(dist)

        [4 3 2 1]
        
    """
    nDim = x.shape[0]
    xC = (x - xI.reshape((nDim, 1)))
    dist = numpy.max(numpy.abs(xC), axis=0)
    return dist
#_______________________________________________________________________________
