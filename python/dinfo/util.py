"""
Utility functions

Create state space model with delays 

Contain

array2D - convert into 2D array

getListOfCases - combined lists into an exhaustive list of cases 

getT - get the extension $x_{t-p}^t$ at order p
getTM1 - get the extension $x_{t-p}^{t-1}$ at order p
getTList - get the extension $x_S$ with $S$ list of retained past samples 

mse - mean squared error
msecoef - mse between vectors

preprocessRedCent - preprocess data by reducing and centering   

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
def array2D(x):
    """Transform 1D array into 2D array
    
    Syntax 
    
    x = array2D(x)
    
    Input
    
    x: float, list, 1Darray or nDArray 
    
    Output
    
    x: 2DArray or nothing
    
    Example
    
    >>> x = numpy.array([1, 2, 3, 4])
    >>> x = util.array2D(x)
    >>> print(x)
    
        [[1 2 3 4]]
    
    """
    if type(x) in (list, int, float): 
        x = numpy.array(x)
    if (x.ndim == 0): 
        x = x.reshape(1, 1)
    elif (x.ndim == 1): 
        x = x.reshape(1, x.shape[0])
    elif (x.ndim == 2): 
        pass
    elif (x.ndim > 2):
        print("ndim > 2")
    return x
#_______________________________________________________________________________
#_______________________________________________________________________________
def getListOfCases(tupleListParam): 
    """Get a list of all cases combining list in a tuple
    
    Syntax 
    
    listOfCases = util.getListOfCases(tupleListParam)
    
    Input
    
    tupleListParams: tuple of nList lists 
    
    Output
    
    listOfCases: (nList, nL1 * ... * nLnList), list of list of cases
    
    Example
    
    >>> param = ([1, 2, 3, 4], [11., 12.], [21, 22, 23])
    >>> listOfCases = util.getListOfCases(param)
    >>> print(listOfCases)
        [[  1.   2.   3.   4.   1.   2.   3.   4.   1.   2.   3.   4.   1.   2.
            3.   4.   1.   2.   3.   4.   1.   2.   3.   4.]
         [ 11.  11.  11.  11.  12.  12.  12.  12.  11.  11.  11.  11.  12.  12.
           12.  12.  11.  11.  11.  11.  12.  12.  12.  12.]
         [ 21.  21.  21.  21.  21.  21.  21.  21.  22.  22.  22.  22.  22.  22.
           22.  22.  23.  23.  23.  23.  23.  23.  23.  23.]]

    """
    nList = len(tupleListParam)
    nParam = numpy.zeros((nList, ))
    for i in range(nList): 
        nParam[i] = len(tupleListParam[i])
    # number of cases 
    nCases = int(numpy.prod(nParam))
    listOfCases = numpy.zeros((nList, nCases))
    p = 1
    for i in range(nList): 
        # print("i: " + str(i))
        for j in range(nCases): 
            k = int(numpy.mod(j / p, nParam[i])) 
            # print(k)
            listOfCases[i, j] = tupleListParam[i][k]
        p *= nParam[i]
    return listOfCases
#_______________________________________________________________________________
#_______________________________________________________________________________
def getTM1(x, p): 
    """Compute the extension x^{t-1} at order p
    
    Syntax
    
    xTM1 = util.getTM1(x, p)
    
    Input
    
    x: (nDim, nObs)
    p: int, order of the extension
    
    Output
    
    xTM1: (p * nDim, nObs)

    Description
    
    $$ \\vec{x}(n) = (x1(n), x2(n), ..., xd(n))^t $$
    $$ \\vec{xTM1}(n, p) = (x1(n - 1), ..., x1(n - p), x2(n - 1), ... 
        xd(n - p))^t $$
            
    Example 
    
    >>> x = numpy.array([1, 2, 3, 4, 5])
    >>> print(util.getTM1(x, 2))
     
    [[ 0.  0.  2.  3.  4.]
     [ 0.  0.  1.  2.  3.]]
    
    Example
    
    >>> x = numpy.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
    >>> print(util.getTM1(x, 2))
   
    [[  0.   0.   2.   3.   4.]
     [  0.   0.   1.   2.   3.]
     [  0.   0.  12.  13.  14.]
     [  0.   0.  11.  12.  13.]]

    """
    if (x.ndim == 1): 
        nDim = 1
        nObs = len(x)
        x = numpy.array([x])
    elif (x.ndim > 1): 
        (nDim, nObs) = x.shape
    else:
        return None
    assert(p < nObs - 1)
    xTM1 = numpy.zeros((p * nDim, nObs))
    for iObs in range(p , nObs):
        for iDim in range(nDim): 
            for iP in range(p): 
                i1 = (p * iDim) + iP
                i2 = iObs - iP - 1
                xTM1[i1, iObs] = x[iDim, i2]
    return xTM1
#_______________________________________________________________________________
#_______________________________________________________________________________
def getT(x, p): 
    """Compute the extension x^{t} at order p
    
    Syntax
    
    xT = util.getT(x, p)
    
    Input
    
    x: (nDim, nObs)
    p: int, order of the extension
    
    Output
    
    xT: ((p + 1) * nDim, nObs)
    
    Description
    
    $$ \\vec{x}(n) = (x1(n), x2(n), ..., xd(n))^t $$
    $$ \\vec{xT}(n, p) = (x1(n), ..., x1(n-p), x2(n), ..., xd(n-p))^t $$

    Example 
    
    >>> x = numpy.array([1, 2, 3, 4, 5])
    >>> print(util.getT(x, 2))
  
    [[ 0.  0.  3.  4.  5.]
     [ 0.  0.  2.  3.  4.]
     [ 0.  0.  1.  2.  3.]]
 
    Example
    
    >>> x = numpy.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
    >>> print(util.getT(x, 2))

    [[  0.   0.   3.   4.   5.]
     [  0.   0.   2.   3.   4.]
     [  0.   0.   1.   2.   3.]
     [  0.   0.  13.  14.  15.]
     [  0.   0.  12.  13.  14.]
     [  0.   0.  11.  12.  13.]]

    """
    if (x.ndim == 1): 
        nDim = 1
        nObs = len(x)
        x = numpy.array([x])
    elif (x.ndim > 1): 
        (nDim, nObs) = x.shape
    else:
        return None
    assert(p < nObs)
    xT = numpy.zeros(((p + 1) * nDim, nObs))
    for iObs in range(p, nObs):
        for iDim in range(nDim): 
            for iP in range(p + 1): 
                i1 = ((p + 1) * iDim) + iP
                i2 = iObs - iP
                xT[i1, iObs] = x[iDim, i2]
    return xT
#_______________________________________________________________________________
#_______________________________________________________________________________
def getTList(x, t): 
    """Compute the extension x_{t-t0, t-t1, ..., t-tNT} at time t
    
    Syntax
    
    xT = util.getTList(x, t)
    
    Input
    
    x: (nDim, nObs)
    t: list of delays with nT values
    
    Output
    
    xT: (nT * nDim, nObs)
    
    Description
    
    $$ \\vec{x}(n) = (x1(n), x2(n), ..., xd(n))^t $$
    $$ \\vec{xT}(n) = (x1(n-t0), ..., x1(n-tNT), x2(n-t0), ..., xd(n-tNT))^t $$
    This is a generalization of getT and getTM1
    getT(x, p) = getTList(x, range(0, p+1))
    getTM1(x, p) = getTList(x, range(1, p+1))

    Example 
    
    >>> x = numpy.array([1, 2, 3, 4, 5])
    >>> print(util.getTList(x, [1, 3]))
  
        [[ 0.  0.  0.  3.  4.]
         [ 0.  0.  0.  1.  2.]]
 
    Example
    
    >>> x = numpy.array([[1, 2, 3, 4, 5, 6, 7], [11, 12, 13, 14, 15, 16, 17]])
    >>> print(util.getTList(x, [0, 2, 4]))
  
        [[  0.   0.   0.   0.   5.   6.   7.]
         [  0.   0.   0.   0.   3.   4.   5.]
         [  0.   0.   0.   0.   1.   2.   3.]
         [  0.   0.   0.   0.  15.  16.  17.]
         [  0.   0.   0.   0.  13.  14.  15.]
         [  0.   0.   0.   0.  11.  12.  13.]]
  
    See also util.getT, util.getTM1
    """
    x = array2D(x)
    (nDim, nObs) = x.shape
    nT = len(t)
    xT = numpy.zeros((nT * nDim, nObs))
    iTMax = max(t)
    for iObs in range(iTMax, nObs):
        for iDim in range(nDim): 
            for iT in range(nT): 
                i1 = nT * iDim + iT
                i2 = iObs - t[iT]
                xT[i1, iObs] = x[iDim, i2]
    return xT
#_______________________________________________________________________________
#_______________________________________________________________________________
def mse(x, xHat): 
    """Compute the mean squared error between x and xHat
    
    Syntax
    
    s = util.mse(x, xHat)
    
    Input
    
    x: (nObs, )
    xHat: (nObs, )
    
    Output
    
    s: float
    
    Example
    
    >>> x = numpy.array([1., 2., 3., 4.])
    >>> xHat = numpy.array([1.1, 1.9, 3.2, 3.9])
    >>> s = util.mse(x, xHat)
    >>> print(s)
    
        0.0175
    
    """
    nObs = len(x)
    s = 0. 
    for i in range(nObs): 
        s += (x[i] - xHat[i]) ** 2
    s /= nObs
    return s
#_______________________________________________________________________________
#_______________________________________________________________________________
def msecoef(x): 
    """Compute the mean squared errors between vectors in x. 
    
    Syntax
    
    E = util.msecoef(x)
    
    Input
    
    x: (nDim, nObs)
    
    Output 
    
    E: (nDim, nDim)
    
    Example 
    
    >>> x = numpy.array([[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7]])
    >>> E = util.msecoef(x)
    >>> print(E)
    
        [[ 0.  1.  9.]
         [ 1.  0.  4.]
         [ 9.  4.  0.]]

    """
    (nDim, nObs) = x.shape
    E = numpy.zeros((nDim, nDim))
    for iDim in range(nDim):
        E[iDim, iDim] = 0
        for jDim in range(iDim + 1, nDim): 
            E[iDim, jDim] = mse(x[iDim, :], x[jDim, :])
    E = E + E.T
    return E
#_______________________________________________________________________________
#_______________________________________________________________________________
def preprocessRedCent(x): 
    """Reduce and center an array of vectors. 
    
    Syntax
    
    xC = util.preprocessRedCent(x)
    
    Input
    
    x: (nDim, nObs)
    
    Output
    
    xC: (nDim, nObs)
    
    Example 
    
    >>> x = numpy.array([[1., 2., 3., 4., 5.]])
    >>> xC = util.preprocessRedCent(x)
    >>> print(xC)
    
    [[-1.41421356 -0.70710678  0.          0.70710678  1.41421356]]
    
    """
    nDim = x.shape[0]
    xC = numpy.empty(x.shape)
    for i in range(nDim): 
        xMean = numpy.mean(x[i, :])
        xSTD = numpy.std(x[i, :])
        xC[i, :] = (x[i, :] - xMean) / xSTD
    return xC
#_______________________________________________________________________________
