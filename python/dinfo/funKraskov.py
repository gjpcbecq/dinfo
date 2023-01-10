"""
Estimation of entropy and mutual information using Kraskov's estimators. 

Contain

h - entropy
hc - conditional entropy
mi - mutual information
mi1 - mutual information with method 1
mi2 - mutual information with method 2

Reference

Kraskov, A.; Stogbauer, H. and Grassberger, P., ``Estimating mutual information'', Physical Review E,  APS, 69, 066138, 2004.

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
import kNN
import distance
import numpy
from scipy.special import psi 
#_______________________________________________________________________________
def h(x, k=1): 
    """Compute entropy h(x)
    
    Syntax
    
    hX = h(x, k=1)
    
    Input
    
    x: (nDim, nObs) 
    k: int, number of neighbors
    
    Output
    
    hX: float
    
    Description
    
    This is actually the Leonenko entropy estimator, see Kraskov 2004.
    $$ \hat{H}(X) = \psi(k) + \psi(N) + \log{c_d} + 
        \frac{d}{N} \, \sum_{i = 1} ^{N} \log{\epsilon_i} $$
    
    Example

    >>> numpy.random.seed(1)
    >>> x = 2. * numpy.random.rand(1, 100)
    >>> print("h exp: " + str(funKraskov.h(x, 10) / numpy.log(2)) + " bits")
    >>> print("h theory: 1 bit")
    
        h exp: 1.03744992955 bits
        h theory: 1 bit
        
    Example

    >>> numpy.random.seed(1)
    >>> x = 3. * numpy.random.rand(4, 100)
    >>> print("h exp: " + str(funKraskov.h(x, 10)))
    >>> nDim = 4
    >>> hTh = nDim * numpy.log(3)
    >>> print("h theory: " + str(hTh))
    
        h exp: 5.23879595487
        h theory: 4.39444915467
        
    Example
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.randn(3, 100)
    >>> print("h exp: " + str(funKraskov.h(x, 10)))
    >>> detC = 1
    >>> nDim = 3
    >>> hTh = 1. / 2. * numpy.log((2 * numpy.pi * numpy.e) ** nDim)
    >>> print("h theory: " + str(hTh))
    
        h exp: 4.00058808611
        h theory: 4.25681559961    
    
    """
    from scipy.special import (gamma as spe_gamma, psi as spe_psi)
    if (x.ndim == 1): 
        x = numpy.array([x])
    (d, nObs) = x.shape
    cD = numpy.pi ** (d / 2.) / spe_gamma(1. +  d / 2.)
    epsilon = kNN.dist(x, k)
    # see Eq 20 Kraskov2004
    hX1 = - spe_psi(k) + spe_psi(nObs) + numpy.log(cD)
    hX2 = float(d) / nObs * numpy.sum(numpy.log(epsilon[-1, :]))
    hX = hX1 + hX2
    return hX
#_______________________________________________________________________________
#_______________________________________________________________________________
def hc(x, y, k=1): 
    """ Compute conditional entropy h(x | y) 
    
    Syntax
    
    (hXkY, hX, miXY) = hc(x, y, k=1)
        
    Input
    
    x: nDimX, nObs
    y: nDimY, nObs
    k=1: number of neighbors. 
    
    Output
    
    hXkY: float
    hX: float
    miXY: float
    
    Description
    
    $$ h(X | Y) = h(X) - i(X; Y)  $$
    
    Example
    
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> (hXkY, hX, miXY) = funKraskov.hc(x, y, 20)
    >>> (hYkX, hY, miXY) = funKraskov.hc(y, x, 20)
    >>> print("hXkY: " + str(hXkY))
    >>> print("hYkX: " + str(hYkX))
        
        hXkY: 0.658359320437
        hYkX: 0.6801707802
    
    """
    # 1st method
    """
    xy = numpy.vstack((x, y))
    hXY = h(xy, k)
    hY = h(y, k)
    hXkY = hXY - hY 
    return (hXkY, hXY, hY) 
    """
    # 2nd method
    hX = h(x, k)
    miXY = mi(x, y, k)
    hXkY = hX - miXY 
    return (hXkY, hX, miXY) 
#_______________________________________________________________________________
#_______________________________________________________________________________
def mi1(x, y, k=1, metric="Euclidean"): 
    """Compute mutual information mi(x; y) using Kraskov's first method 
    
    Syntax
    
    (miXY, epsilon, nX, nY) = mi1(x, y, k=1, metric="Euclidean") 
    
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    k=1: int, number of neighbors
    metric="Euclidean": str, "Euclidean" or "max"
    
    Output
    
    miXY: float
    epsilon: (3, nObs), distance to Z, X, and Y
    nX: (nObs, ), number of samples near X 
    nY: (nObs, ), number of samples near Y
    
    Description
    
    See reference for details 

    Example
    
    >>> numpy.random.seed(1)
    >>> x = 3. * numpy.random.rand(3, 100) 
    >>> y = 1. / 2. * numpy.random.rand(2, 100)
    >>> (miXY, epsilon, nX, nY) = funKraskov.mi1(x, y, 10) 
    >>> print(miXY)
    >>> print("0 expected") 
    
        -1.24344978758e-14
        0 expected
        
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> (miXY, epsilon, nX, nY) = funKraskov.mi1(x, y, 10) 
    >>> print(miXY)
    
        0.744012468135

    """
    if (x.ndim == 1): 
        nObs = x.shape[0]
        x = numpy.reshape(x, (1, nObs))
    if (y.ndim == 1): 
        nObs = y.shape[0]
        y = numpy.reshape(y, (1, nObs))
    (nDimX, nObs) = x.shape
    (nDimY, nObs) = y.shape
    dX = numpy.zeros((nObs,))
    dY = numpy.zeros((nObs,))
    if (metric == "Euclidean"): 
        funDist = distance.Euclid_xXI
    elif (metric == "max"): 
        funDist = distance.max_xXI
    epsilon = numpy.zeros((3, nObs)) 
    # we remove one point for counting due to the 0 distance from xi to himself. 
    nX = numpy.zeros((nObs, )) - 1
    nY = numpy.zeros((nObs, )) - 1
    dZ = numpy.zeros((nObs, ))
    (IZ, IX, IY) = (0, 1, 2)
    for iObs in range(nObs): 
        dX = funDist(x, x[:, iObs])
        dY = funDist(y, y[:, iObs])
        for jObs in range(nObs): 
            dZ[jObs] = numpy.max([dX[jObs], dY[jObs]])
        iZSorted = numpy.argsort(dZ)
        iKNN = iZSorted[k]
        epsilon[IZ, iObs] = dZ[iKNN]
        epsilon[IX, iObs] = dZ[iKNN]
        epsilon[IY, iObs] = dZ[iKNN]
        for i in range(nObs): 
            if (dX[i] < epsilon[IX, iObs]): 
                nX[iObs] += 1
            if (dY[i] < epsilon[IY, iObs]):
                nY[iObs] += 1
    # Eq. 8, Kraskov 2004
    psiN = numpy.mean(psi(nX + 1) + psi(nY + 1))
    miXY = psi(k) - psiN + psi(nObs)
    return (miXY, epsilon, nX, nY)
#_______________________________________________________________________________
#_______________________________________________________________________________
def mi2(x, y, k=1, metric="Euclidean"): 
    """Compute mutual information mi(x; y) using 2nd method of Kraskov

    
    Syntax
    
    (miXY, epsilon, nX, nY) = mi2(x, y, k=1, metric="Euclidean") 
    
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    k=1: int, number of neighbors
    metric="Euclidean": str, "Euclidean" or "max"
    
    Output
    
    miXY: float
    epsilon: (3, nObs), distance to Z, X, and Y
    nX: (nObs, ), number of samples near X 
    nY: (nObs, ), number of samples near Y
    
    Description
    
    See reference for details 

    Example

    >>> numpy.random.seed(1)
    >>> x = 3. * numpy.random.rand(3, 100) 
    >>> y = 1. / 2. * numpy.random.rand(2, 100)
    >>> (miXY, epsilon, nX, nY) = funKraskov.mi2(x, y, 10) 
    >>> print(miXY)
    >>> print("0 expected") 
    
        0.0113562286978
        0 expected
        
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> (miXY, epsilon, nX, nY) = funKraskov.mi2(x, y, 10) 
    >>> print(miXY)

        0.765760046034

    """
    if (x.ndim == 1): 
        nObs = x.shape[0]
        x = numpy.reshape(x, (1, nObs))
    if (y.ndim == 1): 
        nObs = y.shape[0]
        y = numpy.reshape(y, (1, nObs))
    (nDimX, nObs) = x.shape
    (nDimY, nObs) = y.shape
    dX = numpy.zeros((nObs,))
    dY = numpy.zeros((nObs,))
    if (metric == "Euclidean"): 
        funDist = distance.Euclid_xXI
    elif (metric == "max"): 
        funDist = distance.max_xXI
    epsilon = numpy.zeros((3, nObs))
    # we remove one point for counting due to the 0 distance from xi to himself. 
    nX = numpy.zeros((nObs, )) - 1
    nY = numpy.zeros((nObs, )) - 1
    dZ = numpy.zeros((nObs, ))
    (IZ, IX, IY) = (0, 1, 2)
    for iObs in range(nObs): 
        dX = funDist(x, x[:, iObs])
        dY = funDist(y, y[:, iObs])
        for jObs in range(nObs): 
            dZ[jObs] = numpy.max([dX[jObs], dY[jObs]])
        iZSorted = numpy.argsort(dZ)
        iKNN = iZSorted[k]
        epsilon[IZ, iObs] = dZ[iKNN]
        epsilon[IX, iObs] = numpy.max(dX[iZSorted[:k+1]])
        epsilon[IY, iObs] = numpy.max(dY[iZSorted[:k+1]])
        for i in range(nObs): 
            if (dX[i] <= epsilon[IX, iObs]): 
                nX[iObs] += 1
            if (dY[i] <= epsilon[IY, iObs]):
                nY[iObs] += 1
    # Eq. 9, Kraskov 2004
    psiNxNy = numpy.mean(psi(nX) + psi(nY))
    miXY = psi(k) - 1. / k - psiNxNy + psi(nObs)
    return (miXY, epsilon, nX, nY)
#_______________________________________________________________________________
#_______________________________________________________________________________
def mi(x, y, k=1, metric='Euclidean'): 
    """Compute mutual information mi(x; y) 
        
    Syntax
    
    miXY = mi1(x, y, k=1, metric="Euclidean") 
    
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    k=1: int, number of neighbors
    metric="Euclidean": str, "Euclidean" or "max"
    
    Output
    
    miXY: float
    
    Description
    
    wrapper to funKraskov.mi1
    
    Example
    
    >>> numpy.random.seed(1)
    >>> x = 3. * numpy.random.rand(3, 100) 
    >>> y = 1. / 2. * numpy.random.rand(2, 100)
    >>> miXY = funKraskov.mi(x, y, 10) 
    >>> print(miXY)
    >>> print("0 expected")
    
        -1.24344978758e-14
        0 expected

    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> (miXY, epsilon, nX, nY) = funKraskov.mi1(x, y, 10) 
    >>> print(miXY)
    
        0.744012468135

    See also funKraskov.mi1
    
    """
    miXY = mi1(x, y, k, metric)[0]
    return miXY
#_______________________________________________________________________________
