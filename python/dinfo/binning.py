"""
Functions used for binning  

Contain

findBin - Find for x the index of the bin given a set of thresholds
h - entropy 
hc - conditional entropy
mi - mutual information
mic - conditional or partial mutual information
prob - probability in bins

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
import util
#_______________________________________________________________________________
#_______________________________________________________________________________
def findBin(th, x):
    """Find the index of the bin given a set of thresholds such that th[i] <= x < th[i + 1]
    
    Syntax
    
    iBin = findBin(th, x)

    Input
    
    th: (nTh, ) array of thresholds
    x: float, value to classify

    Output

    iBin: int, index of the bin

    Description
    
    first bin is defined by x < th[i] returns 0
    last bin is defined by x >= th[-1] returns len(th)
    
    Use numpy.digitize instead 
    
    Example
    
    >>> th = numpy.array([1.5, 2., 3.])
    >>> f1 = numpy.vectorize(binning.findBin, excluded={0})
    >>> print(f1(th, [0, 1.5, 1.6, 2., 2.5, 3, 3.1]))
    
        [0 1 1 2 2 3 3]
    
    See also numpy.digitize 
    """
    iBin = 0
    while True: 
        if (th[iBin] <= x): 
            if (iBin == (th.size - 1)):
                iBin = th.size; 
                break; 
            else: 
                iBin += 1
        else:
            break; 
    return iBin
#_______________________________________________________________________________
#_______________________________________________________________________________
def prob(x, nBin=2): 
    """Compute probability in bins: [-inf, th[1][, [th[i], th[i+1][, [th[nBin], inf[
    
    Syntax
    
    (p, stepX, thX, xS) = prob(x, nBin=2) 
    
    Input
    
    x: (nDim, nObs), multivariate
    nBin=2: int, number of bins
    
    Output
    
    p: (nBin ** nDim, ), probability to be in bins
    stepX: (nDim, ), width of the bins
    thX: (nDim, nBin + 1), thresholds
    xS: (nDim, nObs), sorted values
    
    Description
    
    For each dimension: 
    bin_0: -inf <= x < th[0]
    bin_1: th[0] <= x < th[1]
    ...
    bin_nBin: th[nBin] <= x < inf

    th[k] = min(x) + k * ptp(x) / nBin
    mid[k] = min(x) + (k + 1/2) * ptp(x) / nBin 
    mid[k] = (th[k] + th[k + 1]) / 2
    th[k] = mid[k] + 1/2 * ptp(x) / nBin
    
    -inf < xmin=th[0] < mid[0] < ... 
        ... < th[nBin] < mid[nBin] < th[nBin+1] = xmax < inf
    
    Indices in p are obtained with j = xBin[iDim, iObs] * (nBin ** iDim)
    
    Example
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.rand(1, 100)
    >>> (p, stepx, thX, xS) = binning.prob(x, 3)
    >>> print((p, stepx, thX))
    
        (array([ 0.35,  0.29,  0.36]), array([ 0.32958224]), array([[  1.14374817e-04,   3.29696613e-01,   6.59278851e-01,
          9.88861089e-01]]))
        
    """
    if x.ndim == 1: 
        x = numpy.array([x])
    (nDim, nObs) = x.shape
    xS = numpy.empty(x.shape, dtype='int')
    # need nBin - 1 thresholds to have nBin bins per dimension.
    nTh = nBin + 1
    thX = numpy.zeros((nDim, nTh), dtype='float')
    p = numpy.zeros((nBin ** nDim, ))
    xBin = numpy.zeros(x.shape, dtype='int')
    for iDim in range(nDim): 
        xS[iDim, :] = numpy.argsort(x[iDim, :])
        minX = x[iDim, xS[iDim, 0]]
        maxX = x[iDim, xS[iDim, -1]]
        thX[iDim, :] = numpy.linspace(minX, maxX, nBin + 1)
        xBin[iDim, :] = numpy.digitize(x[iDim, :], thX[iDim, 1:nBin])
    # index j of bin is given by: j = \sum_i=1^d x_{i-1} \, b ^ (i - 1)
    # with b number of bins on marginal
    for iObs in range(nObs): 
        j = 0
        for iDim in range(nDim): 
            j += xBin[iDim, iObs] * (nBin ** iDim)
        p[j] += 1 
    stepX = numpy.empty((nDim,))
    for iDim in range(nDim): 
        stepX[iDim] = thX[iDim, 1] - thX[iDim, 0]
    p /= nObs
    return (p, stepX, thX, xS)   
#_______________________________________________________________________________
#_______________________________________________________________________________
def h(x, nBin=2, mode="marginal"): 
    """Compute entropy h(x)
    
    Syntax
    
    hX = h(x, nBin=3, mode="marginal") 
    
    Input

    x: (nDim, nObs) or (nObs, )
    nBin=3: int, number of bins
    mode="marginal": nBin is considered for marginals
     "total": nBin is the total number of bins. 
     nBin marginal is adjusted to nBin = ceil(nBin ^ (1 / nDim))
    
    Output
    
    hX: float, entropy of X

    Description
    
    hX is given in nats, divide by log(2) to have it in bits. 

    Example
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.randn(3, 100)
    >>> print("h exp: " + str(binning.h(x, 10)))
    >>> print("h theory: " + str(model.GaussianH(numpy.eye(3))))
    
        h exp: 2.25291146641
        h theory: 4.25681559961
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1) 
    >>> print(binning.h(x))
    
        1.49800633435
        
    """
    if (x.ndim == 1): 
        x = numpy.array([x])
    (nDim, nObs) = x.shape
    if (mode == "total"): 
        nBin = numpy.floor(numpy.power(nBin, 1. / nDim))
        if (nBin < 1): 
            nBin = 1
    (p, stepX, thX, xS) = prob(x, nBin)
    vol_hypercube = numpy.prod(stepX)
    hX = 0
    for iP in range(len(p)): 
        if (p[iP] > 0): 
            hX -= p[iP] * numpy.log(p[iP])
    hX += numpy.log(vol_hypercube)
    return hX 
#_______________________________________________________________________________
#_______________________________________________________________________________
def hc(x, y, nBin=2, mode="marginal"): 
    """Compute conditional entropy h(x | y) 
    
    Syntax
    
    (hXKY, hXY, hY) = hc(x, y, nBin=2, mode="marginal")

    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    nBin=2: int, number of bins
    mode="marginal": string, "marginal" or "total"
    
    Output
    
    hXKY: float 
    hXY: float
    hY: float

    Description
    
    h(x | y) = h(x, y) - h(y)
    
    Example

    >>> numpy.random.seed(1)
    >>> x = 3 * numpy.random.rand(3, 100)
    >>> y = 0.5 * numpy.random.rand(2, 100)
    >>> print("h(X|Y): {0[0]}, h(X, Y): {0[1]}, h(Y): {0[2]}".format(binning.hc(x, y)))
    >>> print("hTh(X|Y): {0}, hTh(X, Y): {1}, hTh(Y): {2}".format(3. * numpy.log(3.), 3. * numpy.log(3.) - 2. * numpy.log(2.), - 2. * numpy.log(2.)))
        
        h(X|Y): 3.12940386691, h(X, Y): 1.68838081322, h(Y): -1.44102305369
        hTh(X|Y): 3.295836866, hTh(X, Y): 1.90954250488, hTh(Y): -1.38629436112
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1) 
    >>> print(binning.hc(x, y, 10))

        (0.37651601115819533, 1.6717667303785873, 1.2952507192203919)
        
    """
    if (x.ndim == 1):  
        x = numpy.array([x])
    if (y.ndim == 1):
        y = numpy.array([y])
    assert(x.shape[1] == y.shape[1]) # nObs the same
    xy = numpy.vstack((x, y))
    hXY = h(xy, nBin, mode)
    hY = h(y, nBin, mode)
    hXKY = hXY - hY 
    return (hXKY, hXY, hY)
#_______________________________________________________________________________
#_______________________________________________________________________________
def mi(x, y, nBin=2, mode="marginal"): 
    """Compute mutual information mi(x; y)
    
    Syntax
    
    (miXY, hX, hY, hXY) = mi(x, y, nBin=2, mode="marginal")

    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    nBin=2: int, number of bins
    mode="marginal": string, "marginal" or "total"

    Output
    
    miXY: float 
    hX: float
    hY: float
    hXY: float

    Description
        
    mi(x; y) = h(x) + h(y) - h(x, y)

    Example

    >>> numpy.random.seed(1)
    >>> x = 3 * numpy.random.rand(100)
    >>> y = 0.5 * numpy.random.rand(100)
    >>> print(binning.mi(x, y, 10))
   
        (0.41147853584901606, 1.0640457875854379, -0.79559996901687358, -0.14303271728045175)

    Example

    >>> numpy.random.seed(1)
    >>> x = 3 * numpy.random.rand(3, 100)
    >>> y = 1. / 2. * numpy.random.rand(2, 100)
    >>> print("mi: {0[0]}, hX: {0[1]}, hY: {0[2]}, hXY: {0[3]}".format(binning.mi(x, y)))
    >>> print("mi: {0[0]}, hX: {0[1]}, hY: {0[2]}, hXY: {0[3]}".format(mi(x, y, 10)))
    >>> print("miTh: 0, hXTh: {0}, hYTh: {1}, hXYTh: {2}".format(3. * numpy.log(3.), 2. * numpy.log(0.5), 3. * numpy.log(3.) + 2. * numpy.log(0.5)))
    
        mi: 0.0945706746238, hX: 3.22397454153, hY: -1.44102305369, hXY: 1.68838081322
        mi: 3.99153166569, hX: 0.893391207085, hY: -1.98645894777, hXY: -5.08459940638
        miTh: 0, hXTh: 3.295836866, hYTh: -1.38629436112, hXYTh: 1.90954250488
    
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1) 
    >>> (miXY, hXY, hX, hY) = binning.mi(x, y)
    >>> print((miXY, hXY, hX, hY))
    
        (0.20292734500561016, 1.4980063343468708, 1.581428882183709, 2.8765078715249697)
    
    """
    if (x.ndim == 1):  
        x = numpy.array([x])
    if (y.ndim == 1):
        y = numpy.array([y])
    assert(x.shape[1] == y.shape[1]) # nObs the same
    hX = h(x, nBin, mode)
    hY = h(y, nBin, mode)
    xy = numpy.vstack((x, y))
    hXY = h(xy, nBin, mode)
    miXY = hX + hY - hXY
    return (miXY, hX, hY, hXY)
#_______________________________________________________________________________
#_______________________________________________________________________________
def mic(x, y, z, nBin=2, mode="marginal"): 
    """Compute conditional mutual information h(x; y | z)
    
    Syntax
    
    (micXYKZ, hXZ, hYZ, hXYZ, hZ) = mic(x, y, z, nBin=2, mode="marginal")
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    z: (nDimZ, nObs) or (nObs, )
    
    Output
    
    micXYKZ: float
    hXZ: float
    hYZ: float
    hXYZ: float
    hZ: float

    Description
        
    mi(x; y | z) = h(x, z) + h(y, z) - h(x, y, z) - h(z)
    
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1) 
    >>> (micXYKZ, hXZ, hYZ, hXYZ, hZ) = binning.mic(x, y, z)
    >>> print((micXYKZ, hXZ, hYZ, hXYZ, hZ))
    
        (0.20913249548481749, 3.0918086507360538, 3.1178816963647877, 4.4010864000453891, 1.599471451570635)

    """
    xyz = numpy.vstack((x, y, z))
    yz = numpy.vstack((y, z))
    xz = numpy.vstack((x, z))
    hXYZ = h(xyz, nBin, mode)
    hYZ = h(yz, nBin, mode)
    hXZ = h(xz, nBin, mode)
    hZ = h(z, nBin, mode)
    micXYKZ = hXZ + hYZ - hXYZ - hZ
    return (micXYKZ, hXZ, hYZ, hXYZ, hZ)
#_______________________________________________________________________________   
