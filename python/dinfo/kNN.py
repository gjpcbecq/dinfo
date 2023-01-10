"""
k Nearest neighbors functions

Contain

dist - Distances of the k nearest neighbors (kNN)

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
import distance
#_______________________________________________________________________________
def dist(x, k=1, method="Euclidean"):
    """Distances of the k nearest neighbors (kNN)
    
    Syntax
    
    d = dist(x, k=1, method="Euclidean")
    
    Input
    
    x: (nDim, nObs)
    k=1: int, number of neighbors
    method="Euclidean": "Euclidean", "max"
    
    Output
    
    d: (k, nObs), distances to kNN
    
    Description
    
    The algorithm relies on the sort function.  
    For each column j of d, the first line corresponds to the distance of x_j 
    to its first neighbor. 
    The algorithm will return zeros if some samples are the same. 
    To avoid that, a weak noise can be added to x. 
    x = x + alpha * numpy.random.rand(x.shape)
    with alpha set to a small value with respect to the range of x. 
        
    Example 
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.rand(1, 100)
    >>> d = kNN.dist(x, 3)
    >>> numpy.set_printoptions(precision=5, suppress=True)
    >>> print(d[:, 0:7])  
    
        [[ 0.00028  0.00567  0.00276  0.00872  0.00003  0.00601  0.01184]
         [ 0.00217  0.02057  0.01817  0.01109  0.00637  0.00729  0.01643]
         [ 0.00284  0.02592  0.01925  0.01318  0.00748  0.01     0.01819]]
    
    """ 
    if (x.ndim == 1): 
        x = numpy.array([x])
    (nDim, nObs) = x.shape;
    if (k + 1 > nObs): 
        kMax = nObs - 1 
        print("kMax set to nObs - 1")
    else : 
        kMax = k; 
    d = numpy.empty((kMax, nObs))
    if (method == "Euclidean"): 
        funDist = distance.Euclid_xXI
    elif (method == "max"): 
        funDist = distance.max_xXI
    for i in range(nObs):
        dist = funDist(x, x[:, i])
        sortDist = numpy.sort(dist) 
        d[:, i] = sortDist[1:kMax+1] 
    return d
#_______________________________________________________________________________

