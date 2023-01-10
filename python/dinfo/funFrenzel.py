"""
Functions described by S.Frenzel and B. Pompe for computing partial mutual information 

Contain

hN - compute the opposite of the sum of inverse
mi - mutual information
mic - conditional or partial mutual information

Reference

Stefan Frenzel and Bernd Pompe, ``Partial mutual information for coupling analysis of multivariate time series'', PRL 99 (204101): 1--4, 2007

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
import util
#_______________________________________________________________________________
def hN(N): 
    """Compute the opposite of the sum of inverse from 1 to N
    
    Syntax
    
    s = hN(N) 
    
    Input
    
    N: int
    
    Output
    
    s: float
    
    Description
    
    $$ h_N = - \sum_{n=1}^{N} n^{-1} $$
    See Frenzel and Pompe's article for details. 
    
    Example
    
    >>> print(funFrenzel.hN(100))
    
        -5.18737751764

    """
    s = 0.
    if (N > 0): 
        for i in range(1, N + 1): 
            s += 1. / i
    return -s 
#_______________________________________________________________________________
#_______________________________________________________________________________
vHN = numpy.vectorize(hN, otypes=[float])
#_______________________________________________________________________________
#_______________________________________________________________________________
def mic(x, y, z, k=1, metric="Euclidean"): 
    """ Compute conditional mutual information or partial mutual information 
    
    Syntax
    
    (miXYKZ, nXZ, nYZ, nZ) = mic(x, y, z, k=1, metric="Euclidean") 
        
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    z: (nDimZ, nObs)
    k=1: int, number of neighbors
    metric="Euclidean": str, "Euclidean" or "max"
    
    Output
    
    miXYKZ: float 
    nXZ: int
    nYZ: int
    nZ: int
    
    Description
    
    I(X, Y |Z) in the article. 
    $$ \hat{I}(X, Y|Z) = \langle h_{N_{xz}(t)} + h_{N_{yz}(t)} - 
        h_{N_{z}(t)} \rangle - h_{k - 1} $$
    See Frenzel and Pompe's article for details. 
        
    Example 
    
    >>> x = numpy.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
    >>> y = numpy.array([[3, 4, 9], [13, 14, 19], [23, 24, 29]])
    >>> z = numpy.array([[6, 7, 8], [16, 17, 18], [26, 27, 28]])
    >>> print(funFrenzel.mic(x, y, z, 2))
        
        (0.0, 3, 2, 3)
        
    >>> numpy.random.seed(1)
    >>> x = numpy.random.rand(10, 1000)
    >>> y = numpy.random.rand(10, 1000)
    >>> z = numpy.random.rand(3, 1000)
    >>> print(funFrenzel.mic(x, y, z, 10))
    
        (0.007167331102784669, 112, 75, 967)
        
    >>> numpy.random.seed(1)
    >>> from dinfo import model
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(funFrenzel.mic(x, y, z, 10))
        
        (0.7765992965871189, 12, 12, 27)
        
    """
    x = util.array2D(x)
    y = util.array2D(y)
    z = util.array2D(z)
    (nDimX, nObs) = x.shape
    (nDimY, nObs) = y.shape
    (nDimZ, nObs) = z.shape
    dX = numpy.zeros((nObs,))
    dY = numpy.zeros((nObs,))
    dZ = numpy.zeros((nObs,))
    if (metric == "Euclidean"): 
        funDist = distance.Euclid_xXI
    elif (metric == "max"): 
        funDist = distance.max_xXI
    epsilon = numpy.zeros((3, nObs)) 
    # we remove one point for counting due to the 0 distance from xi to himself. 
    sumHN = 0
    dXZ = numpy.zeros((nObs, ))
    dYZ = numpy.zeros((nObs, ))
    dXYZ = numpy.zeros((nObs, ))
    for iObs in range(nObs): 
        dX = funDist(x, x[:, iObs])
        dY = funDist(y, y[:, iObs])
        dZ = funDist(z, z[:, iObs])
        for jObs in range(nObs): 
            dXYZ[jObs] = numpy.max([dX[jObs], dY[jObs], dZ[jObs]])
            dXZ[jObs] = numpy.max([dX[jObs], dZ[jObs]])
            dYZ[jObs] = numpy.max([dY[jObs], dZ[jObs]])
        iXYZSorted = numpy.argsort(dXYZ)
        iXYZKNN = iXYZSorted[k]
        epsilonK = dXYZ[iXYZKNN]
        nXZ = 0
        nYZ = 0
        nZ = 0
        for i in range(nObs): 
            if (dXZ[i] < epsilonK): 
                nXZ += 1
            if (dYZ[i] < epsilonK):
                nYZ += 1
            if (dZ[i] < epsilonK):
                nZ += 1
        # remove 1 from null distance to himself. 
        sumHN += hN(nXZ - 1) + hN(nYZ - 1) - hN(nZ - 1)
    #print((epsilonXZ, epsilonYZ, epsilonZ))
    #print((numpy.min(nXZ), numpy.min(nYZ), numpy.min(nZ)))
    meanHN = float(sumHN) / nObs
    miXYKZ = meanHN - hN(k - 1)
    return (miXYKZ, nXZ, nYZ, nZ)
#_______________________________________________________________________________
#_______________________________________________________________________________
def mi(x, y, k=1, metric="Euclidean"): 
    """ Compute the mutual information 
    
    Syntax
    
    (miXY, nX, nY) = mi(x, y, k=1, metric="Euclidean") 
        
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    k=1: int, number of neighbors
    metric="Euclidean": str, "Euclidean" or "max"
    
    Output
    
    miXY: float 
    nX: int
    nY: int
    
    Description
    
    I(X, Y) in the article. 
    $$ \hat{I}(X, Y) = \langle h_{N_{x}(t)} + h_{N_{y}(t)} \rangle - 
        h_{T-1} - h_{k - 1} $$
    See Frenzel and Pompe's article for details. 
        
    Example 
    
    >>> x = numpy.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
    >>> y = numpy.array([[3, 4, 9], [13, 14, 19], [23, 24, 29]])
    >>> print(funFrenzel.mi(x, y, 2))
        
        (0.0, 3, 2)
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.rand(10, 100)
    >>> y = numpy.random.rand(10, 100)
    >>> print(funFrenzel.mi(x, y, 10))

        (0.003950747666842336, 113, 76)
        
    >>> numpy.random.seed(1)
    >>> from dinfo import model
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(funFrenzel.mi(x, y, 10))

        (0.74401246813501, 24, 19)

    """
    x = util.array2D(x)
    y = util.array2D(y)
    (nDimX, nObs) = x.shape
    (nDimY, nObs) = y.shape
    dX = numpy.zeros((nObs,))
    dY = numpy.zeros((nObs,))
    if (metric == "Euclidean"): 
        funDist = distance.Euclid_xXI
    elif (metric == "max"): 
        funDist = distance.max_xXI
    dX = numpy.zeros((nObs, ))
    dY = numpy.zeros((nObs, ))
    dXY = numpy.zeros((nObs, ))
    sumHN = 0
    # print("nDimX,:{0}, nDimY:{1}, funDist:{2}, nObs:{3}".format(nDimX, 
    #     nDimY, funDist, nObs))
    for iObs in range(nObs): 
        dX = funDist(x, x[:, iObs])
        dY = funDist(y, y[:, iObs])
        for jObs in range(nObs): 
            dXY[jObs] = numpy.max([dX[jObs], dY[jObs]])
        iXYSorted = numpy.argsort(dXY)
        iXYKNN = iXYSorted[k]
        epsilonK = dXY[iXYKNN]
        nX = 0
        nY = 0
        for i in range(nObs): 
            if (dX[i] < epsilonK): 
                nX += 1
            if (dY[i] < epsilonK):
                nY += 1
        # One point is removed due to the null distance from xi to himself. 
        sumHN += hN(nX - 1) 
        sumHN += hN(nY - 1)
    meanHN = float(sumHN) / nObs
    # print((epsilonXZ, epsilonYZ, epsilonZ))
    # print((numpy.min(nXZ), numpy.min(nYZ), numpy.min(nZ)))
    miXY = meanHN - hN(nObs - 1) - hN(k - 1)
    return (miXY, nX, nY)
#_______________________________________________________________________________
