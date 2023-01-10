"""
kernel based methods

Contain 

GaussianGram - Gram matrix with Gaussian kernels
GaussianGramXY - Gram matrix with Gaussian kernels
kernelGaussian - Gaussian kernel
vectorGaussianKernel - Gaussian kernel applied to a vector

LinearGram - Gram matrix with linear kernels
LinearGramXY - Gram matrix with linear kernels
kernelLinear - linear kernel
vectorLinearKernel - Linear kernel applied to a vector

mspe - mean square prediction error 
mspe_crossfold - mean square prediction error with crossfold evaluation
mspe_crossfold_search - mspe with crossfold and optimal search of parameters
optimalKernelLearn - learn optimal kernel
optimalKernelTest - test optimal kernel

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
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky

#_______________________________________________________________________________
# GAUSSIAN KERNEL
#_______________________________________________________________________________
#_______________________________________________________________________________
def kernelGaussian(x, y, beta): 
    """Compute a Gaussian kernel between vector x and y
    
    Syntax 
    
    k = kernelGaussian(x, y, beta)
    
    Input 
    
    x (nDim, )
    y (nDim, )
    beta : scalar,
    
    Output
    
    k: float

    Description
    
    k(x,y) = exp(- ||x - y||^2 / beta ^ 2)

    Example

    >>> x = numpy.array([1, 2, 3])
    >>> print(kernel.kernelGaussian(x, x, 1)) 
        
        1.0 

    Example

    >>> x = numpy.array([1, 2, 3])
    >>> y = numpy.array([2, 3, 4])
    >>> print(kernel.kernelGaussian(x, y, 1)) 
    
        0.0497870683679
        
    Example    
        
    >>> x = numpy.array([1, 2, 3])
    >>> y = numpy.array([2, 3, 4])
    >>> print(kernel.kernelGaussian(x, y, 3))     
        
        0.716531310574
        
    """
    nDim = x.shape[0]
    norm = 0.0
    for i in range(nDim): 
        norm += (x[i] - y[i]) ** 2.0
    k = numpy.exp(- norm / (beta ** 2.0))
    return k
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianGram(A, beta): 
    """Compute a Gram matrix with Gaussian kernel from matrix A
    
    Syntax
    
    G = GaussianGram(A, beta)
        
    Input
    
    A: (nDim, nObs)
    beta: parameter of the Gaussian kernel
    
    Output
    
    G: Gram matrix (nObs, nObs)

    Description
    
    The Gram matrix G is given by: 
    $$ G_{i, j} = k(X_i, X_j) $$ with k kernel function and $X_i = A_{., i}$
 
    Example 
    
    >>> x = numpy.array([[1., 2., 3.], [11., 12., 13.]])
    >>> G = kernel.GaussianGram(x, 2)
    >>> print(floor(G * 100) / 100)

        [[ 1.    0.6   0.13]
         [ 0.6   1.    0.6 ]
         [ 0.13  0.6   1.  ]]
     
    """
    N = A.shape[1]
    G = numpy.zeros((N, N))
    for i in range(N): 
        for j in range(i, N): 
            G[i, j] = kernelGaussian(A[:, i], A[:, j], beta)
            G[j, i] = G[i, j]
    return G
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianGramXY(X, Y, beta): 
    """Compute a Gram matrix with Gaussian kernel from matrix X and Y
    
    Syntax
    
    (GXY, GYX) = GaussianGramXY(X, Y, beta)
    
    Input 
    
    X (nDim, nObsX)
    Y (nDim, nObsY)
    beta is the paramter of the Gaussian kernel
    
    Output
    
    GXY (nObsX, nObsY)
    GYX = GXY.T
    
    Description
    
    X contains the vector in column X_i = X[:, i]. 
    Same for Y
    
    Example
    
    >>> x = numpy.array([[1, 2, 3], [2, 3, 4]])
    >>> y = numpy.array([[1, 2, 3, 4, 5, 1], [2, 3, 4, 5, 6, 2]])
    >>> (GXY, GYX) = kernel.GaussianGramXY(x, y, 2)
    >>> print(floor(GXY * 100) / 100)
    >>> print(floor(GYX * 100) / 100)
    
        [[ 1.    0.6   0.13  0.01  0.    1.  ]
         [ 0.6   1.    0.6   0.13  0.01  0.6 ]
         [ 0.13  0.6   1.    0.6   0.13  0.13]]
        [[ 1.    0.6   0.13]
         [ 0.6   1.    0.6 ]
         [ 0.13  0.6   1.  ]
         [ 0.01  0.13  0.6 ]
         [ 0.    0.01  0.13]
         [ 1.    0.6   0.13]]
    
    """
    nObsX = X.shape[1]
    nObsY = Y.shape[1]
    GXY = numpy.zeros((nObsX, nObsY))
    for i in range(nObsX): 
        for j in range(nObsY): 
            GXY[i, j] = kernelGaussian(X[:, i], Y[:, j], beta)
    GYX = GXY.T
    return (GXY, GYX)
#_______________________________________________________________________________
#_______________________________________________________________________________
def vectorGaussianKernel(wi, w, beta):
    """Compute the vector of kernel function k(wi, w[:, j])
    
    Syntax
    
    k = vectorGaussianKernel(wi, w, beta)
    
    Input 
    
    wi: (nDim, )
    w: (nDim, nObs)
    beta: the kernel parameter
    
    Output
    
    k: (nObs, )
    
    Example
    
    >>> x = numpy.array([[1., 2., 3.], [11., 12., 13.]])
    >>> G = kernel.GaussianGram(x, 2)
    >>> x0 = kernel.vectorGaussianKernel(x[:, 0], x, 2)
    >>> x1 = kernel.vectorGaussianKernel(x[:, 1], x, 2)
    >>> x2 = kernel.vectorGaussianKernel(x[:, 2], x, 2)
    >>> print(G)
    >>> print(x0)
    >>> print(x1)
    >>> print(x2)
    
        [[ 1.          0.60653066  0.13533528]
         [ 0.60653066  1.          0.60653066]
         [ 0.13533528  0.60653066  1.        ]]
        [ 1.          0.60653066  0.13533528]
        [ 0.60653066  1.          0.60653066]
        [ 0.13533528  0.60653066  1.        ]
    
    See also kernel.GaussianGram, kernel.kernelGaussian
    
    """
    nObs = w.shape[1]
    k = numpy.empty(nObs, )
    for j in range(nObs): 
        k[j] = kernelGaussian(wi, w[:, j], beta)
    return k
#_______________________________________________________________________________
#_______________________________________________________________________________
# LINEAR KERNEL 
#_______________________________________________________________________________
#_______________________________________________________________________________
def kernelLinear(x, y, param=0): 
    """Compute a linear kernel between vector x and y
    
    Syntax
    
    k = kernel.kernelLinear(x, y, param=0)
    
    Input
    
    x: (nDim, )
    y: (nDim, )
    param=0: unused but necessary for compatibility with other functions calling
     it
    
    Output 
    
    k: float
    
    Description
    
    k(x, y) = x^t * y
    
    Example
    
    >>> x = numpy.array([1, 2, 3])
    >>> y = numpy.array([2, 3, 4])
    >>> print(kernel.kernelLinear(x, y)) 
    
        20.0

    Example    
        
    >>> x = numpy.array([1, 2, 3])
    >>> y = numpy.array([2, 3, 4])
    >>> print(kernel.kernelLinear(x, y - x))    
        
        6.0
        
    """
    nDim = x.shape[0]
    k = 0.0
    for i in range(nDim): 
        k += x[i] * y[i]
    return k
#_______________________________________________________________________________
#_______________________________________________________________________________
def LinearGram(A, param=0):
    """Compute a Gram matrix with linear kernel from matrix A
    
    Syntax
    
    G = LinearGram(A, param=0)
    
    Input 
    
    A: (nDim, nObs)
    param=0: unused but necessary for compatibility with other functions 
    calling it
     
    Output 
    
    G: (nObs, nObs), Gram matrix 

    Description 
    
    The Gram matrix G is given by: 
    $$ G_{i, j} = k(X_i, X_j) $$ with k kernel function and $X_i = A_{., i}$
    
    Example 
    
    >>> x = numpy.array([[1., 2., 3.], [11., 12., 13.]])
    >>> G = kernel.LinearGram(x)
    >>> print(numpy.floor(G * 100) / 100)
    
        [[ 122.  134.  146.]
         [ 134.  148.  162.]
         [ 146.  162.  178.]]
         
    See also kernel.kernelLinear
    
    """
    N = A.shape[1]
    G = numpy.zeros((N, N))
    for i in range(N): 
        for j in range(i, N): 
            G[i, j] = kernelLinear(A[:, i], A[:, j], param)
            G[j, i] = G[i, j]
    return G
#_______________________________________________________________________________
#_______________________________________________________________________________
def LinearGramXY(X, Y, param=0): 
    """Compute a Gram matrix with linear kernel from matrix X and Y
    
    Syntax
    
    (GXY, GYX) = LinearGramXY(X, Y, param=0)
    
    Input
    
    X: (nDim, nObsX)
    Y: (nDim, nObsY)
    param=0: unused but necessary for compatibility with other functions calling
     it
   
    Output
    
    GXY: (nObsX, nObsY) Gram matrix 
    GYX: (nObsY, nObsX) GYX = GXY.T
    
    Description
    
    X contains the vector in column X_i = X[:, i]. 
    Same for Y
    
    Example
    
    >>> x = numpy.array([[1, 2, 3], [11, 12, 13]])
    >>> y = numpy.array([[1, 2, 3, 4, 5, 1], [11, 12, 13, 14, 15, 11]])
    >>> (GXY, GYX) = kernel.LinearGramXY(x, y)
    >>> print(numpy.floor(GXY * 100) / 100)
    >>> print(numpy.floor(GYX * 100) / 100)
    
        [[ 122.  134.  146.  158.  170.  122.]
         [ 134.  148.  162.  176.  190.  134.]
         [ 146.  162.  178.  194.  210.  146.]]
        [[ 122.  134.  146.]
         [ 134.  148.  162.]
         [ 146.  162.  178.]
         [ 158.  176.  194.]
         [ 170.  190.  210.]
         [ 122.  134.  146.]]
         
    See also kernel.LinearGram, kernel.kernelLinear
    
    """
    nObsX = X.shape[1]
    nObsY = Y.shape[1]
    GXY = numpy.zeros((nObsX, nObsY))
    for i in range(nObsX): 
        for j in range(nObsY): 
            GXY[i, j] = kernelLinear(X[:, i], Y[:, j], param)
    GYX = GXY.T
    return (GXY, GYX)
#_______________________________________________________________________________
#_______________________________________________________________________________
def vectorLinearKernel(wi, w, param=0):
    """Compute the vector of function k(wi, w(.))
    
    Syntax
    
    k = vectorLinearKernel(wi, w, param=0)
    
    Input 
    
    wi: (nDim, )
    w: (nDim, nObs)
    param=0: unused but necessary for compatibility with other functions calling
     it
    
    Output
    
    k is (nObs, )
    
    Example
    
    >>> wi = numpy.array([[1], [11]])
    >>> w = numpy.array([[1, 2, 3], [11, 12, 13]]) 
    >>> beta = 1
    >>> k = kernel.vectorLinearKernel(wi, w - wi)
    >>> print(k)
        
        [  0.  12.  24.]
    
    See also kernel.LinearGram, kernel.kernelLinear
    
    """
    nObs = w.shape[1]
    k = numpy.empty(nObs, )
    for j in range(nObs): 
        k[j] = kernelLinear(wi, w[:, j], param)
    return k
#_______________________________________________________________________________
#_______________________________________________________________________________
# Global function
#_______________________________________________________________________________
#_______________________________________________________________________________
def optimalKernelLearn(xL, wL, kernelMethod, lambda_, param): 
    """Learn optimal parameters for kernel. 
    
    Syntax
    
    alpha = optimalKernelLearn(xL, wL, kernelMethod, lambda_, param)
    
    Input
    
    xL: (nObsL, ) learning targets, nObsL, size of the learning set
    wL: (nDimW, nObsL) learning predictors. 
    kernelMethod: string "Gaussian" or "linear"
    lambda_: float, optimization parameter
    param: (nParam, ) kernel parameters
    
    Output 
    
    alpha: (nObsL, 1) 

    Description
    
    Find the optimal weights for the learning targets xL and the learning 
    predictors wL given lambda and a kernel with its parameters. 

    Example
    
    >>> xL = numpy.array([1, 2, 3]) 
    >>> wL = numpy.array([[2, 3, 4], [1, 1, 1]])
    >>> alpha = kernel.optimalKernelLearn(xL, wL, "Gaussian", 0.1, 1.)
    >>> print(alpha)
    
        [[ 0.60159903]
         [ 0.79742798]
         [ 2.45056725]]
         
    """
    if (kernelMethod in {"Gaussian"}): 
        Gram = GaussianGram
    if (kernelMethod in {"linear"}): 
        Gram = LinearGram
    if (kernelMethod not in {"Gaussian", "linear"}): 
        print("Unknown kernelMethod... please check parameters")
        alpha = numpy.nan
        return alpha
    K = Gram(wL, param)
    K = numpy.matrix(K)
    nObsL = len(xL)
    One = numpy.matrix(numpy.identity(nObsL))
    xL = numpy.matrix(xL).T
    M = (K + lambda_ * One)
    L = cholesky(M)
    alpha = solve(L.T , solve(L, xL))
    # other solution 
    # alpha = M.I * xL
    return alpha
#_______________________________________________________________________________
#_______________________________________________________________________________
def optimalKernelTest(wT, kernelMethod, wL, alpha, param): 
    """Test prediction with optimal parameters using kernels 
    
    Syntax
    
    xTHat = optimalKernelTest(wT, kernelMethod, wL, alpha, param) 
    
    Input
    
    wT: (nDimW, nObsT) tested predictors
    kernelMethod: string "Gaussian" or "linear"
    wL: (nDimW, nObsL) learned predictors
    alpha: (nObsL, ) learned optimal weights 
    param: kernel parameter
    
    Output 
    
    xTHat: (nObsT, ) outputs, targets, predictions
    
    Description
    
    Test learned kernels with optimal weights. 
    
    Example
    
    >>> wL = numpy.array([[2., 3, 4], [1, 1, 1]])
    >>> alpha = numpy.array([[0.6],[0.8],[2.5]]) 
    >>> wT = numpy.array([[2., 3, 4], [1, 1, 1]])
    >>> xT = kernel.optimalKernelTest(wT, "Gaussian", wL, alpha, 1.)
    >>> print(xT)
    >>> wT = numpy.array([[1., 2, 3], [1, 1, 1]])
    >>> xT = kernel.optimalKernelTest(wT, "Gaussian", wL, alpha, 1.)
    >>> print(xT)
    >>> wT = numpy.array([[2., 3, 4], [3, 2, 1]])
    >>> xT = kernel.optimalKernelTest(wT, "Gaussian", wL, alpha, 1.)
    >>> print(xT)
    
        [ 0.94009265  1.94042627  2.80529294]
        [ 0.2356887   0.94009265  1.94042627]
        [ 0.0172184   0.71384293  2.80529294]

    """
    nObsT = wT.shape[1]
    xTHat = numpy.zeros((nObsT, ))
    if (kernelMethod in "Gaussian"): 
        funVectorKernel = vectorGaussianKernel
        GramXY = GaussianGramXY
    if (kernelMethod in "linear"): 
        funVectorKernel = vectorLinearKernel
        GramXY = LinearGramXY
    if (kernelMethod not in {"Gaussian", "linear"}): 
        print("Unknown kernelMethod... please check parameters")
        return nan
    for i in range(nObsT):
        kw = funVectorKernel(wT[:, i], wL, param)
        kw = numpy.matrix(kw)
        xTHatTemp = kw * alpha
        xTHat[i] = xTHatTemp[0, 0]
    # other solution using the Gram Matrix with wT and wL
    """
    kW = GramXY(wT, wL, beta)[0]
    xTHat = numpy.dot(kW.T, alpha)
    """
    return xTHat
#_______________________________________________________________________________
#_______________________________________________________________________________
def mspe(xL, wL, xT, wT, kernelMethod, lambda_, param): 
    """Compute the mean squared prediction error with kernel methods. 

    Syntax
    
    (mspe, xHat) = kernel.mspe(xL, wL, xT, wT, kernelMethod, lambda_, param)

    Input
    
    xL: (nObsL, ) learning set targets
    wL: (nDimW, nObsL) learning set predictors
    xT: (nObsT, ) test set targets
    wT: (nDimW, nObsT) test set predictors
    kernelMethod: "Gaussian" or "linear"
    lambda_:  float, the optimization parameter
    param: kernel parameter
    
    Output 
    
    mspe: float, mean squared prediction error. 
    xTHat: (nObsT, ), the prediction on the test set. 

    Description
    
    Learn on wL, xL
    Test on wT, xT
    w contains the predictors
    x contains the targets

    Example 
    
    >>> xL = numpy.array([1.1 , 1.9, 3.1])
    >>> wL = numpy.array([[1., 2., 3.], [11., 12., 13.]])
    >>> xT = numpy.array([1.1 , 1.9, 3.1])
    >>> wT = numpy.array([[1., 2., 3.], [11., 12., 13.]])
    >>> (mspe, xHat) = kernel.mspe(xL, wL, xT, wT, "Gaussian", 0.1, 10.)
    >>> print((mspe, xHat))
    
        (0.23023282040840856, array([ 1.51802599,  1.98981872,  2.38733926]))
    
    See also kernel.optimalKernelLearn, kernel.optimalKernelTest, util.mse
    
    """
    alpha = optimalKernelLearn(xL, wL, kernelMethod, lambda_, param)
    xTHat = optimalKernelTest(wT, kernelMethod, wL, alpha, param)
    mspe = util.mse(xT, xTHat)  
    return (mspe, xTHat)
#_______________________________________________________________________________
#_______________________________________________________________________________
def mspe_crossfold(x, w, kernelMethod, nFold, lambda_, param): 
    """Compute mean squared prediciton error for kernel with cross validation
    
    Syntax
    
    mmspe = mspe_crossfold(x, w, kernelMethod, nFold, lambda_, param)
    
    Input
    
    x: (nObs, ) 1 dimension, targets 
    w: (nDimW, nObs), predictors
    kernelMethod: "Gaussian" or "linear"
    nFold: int, number of fold the cross validation is applied. 
    lambda_: float, optimization parameter
    param: float, Gaussian parameter
    
    Output
    
    mmspe: float, mean of the mean squared error over the nFold crossvalidation
    
    Description
    
    x is a set of target and w a set of predictors. 
    x is cut into nFold parts such that : 
    [x[0] ... part[0] ...][... part[i] ... ][... part[nFold-1] ... x[n-1]]
    The same is done for w. 
    Learning is done on the nFold-1 parts and tested on the last one. 
    The learning and test sets are exchanged nFold times.  
    The signal must be stationary on each part to ensure appropriate learning and good performances. 
    
    Example
    
    >>> numpy.random.seed(1)
    >>> x = numpy.arange(1, 101)
    >>> w = numpy.vstack((numpy.arange(0, 100), numpy.ones((100, ))))
    # need a stationary signal: a permutation is done on the whole set. 
    >>> iPerm = numpy.random.permutation(100) 
    >>> x = x[:, iPerm];
    >>> w = w[:, iPerm];
    >>> mmspe = numpy.zeros((6, 5));
    >>> listLambda = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    >>> listBeta = [1e1, 1e2, 1e3, 1e4, 1e5]
    >>> for (i, lambda_) in enumerate(listLambda): 
            for (j, beta) in enumerate(listBeta): 
                mmspe[i, j] = kernel.mspe_crossfold(x, w, "Gaussian", 10, 
                    lambda_, beta)
    >>> print(numpy.round(log(mmspe), 1))
    
        [[ 6.2  3.1  6.6  6.9  6.9]
         [ 6.   1.8  5.1  6.8  6.9]
         [ 6.   0.6  1.5  6.6  6.9]
         [ 6.  -1.5 -2.9  5.1  6.8]
         [ 6.  -2.6 -6.   1.5  6.6]
         [ 6.  -3.2 -6.9 -3.   5.1]]
 
    See also kernel.mspe, kernel.mspe_crossfold_search
    
    """
    def xFoldGetIL(nObs, nFold):
        nSet = nObs // nFold
        iL = []
        k = 0
        for i in range(nFold):
            iL.append(k) 
            k += nSet
        return (iL, nSet)

    def xFoldGetA(nSet, nFold): 
        A = range(0, nSet * nFold)
        return A

    def xFoldGetL(iL, nSet): 
        L = range(iL, iL + nSet)
        return L

    def xFoldGetT(L, A): 
        T = list(set(A).difference(set(L)))
        return T
    
    assert(nFold > 1)
    nObs = x.shape[0]
    assert(nFold < nObs + 1)
    # print(nObs, nFold)
    (iL, nSet) = xFoldGetIL(nObs, nFold)
    A = xFoldGetA(nSet, nFold)
    mmspe = 0
    for i in range(nFold):
        L = xFoldGetL(iL[i], nSet)
        T = xFoldGetT(L, A)
        # print((A[0], L[0], T[0]))
        xL = x[L]
        wL = w[:, L]
        xT = x[T]
        wT = w[:, T]
        (mmspeT, xHatT) = mspe(xL, wL, xT, wT, kernelMethod, lambda_, param)
        # print((mmspeT, xHatT))
        mmspe += mmspeT
    mmspe /= nFold
    return mmspe
#_______________________________________________________________________________
#_______________________________________________________________________________
def mspe_crossfold_search(x, w, kernelMethod, nFold, listLambda, tupleParam): 
    """Compute mean squared prediciton error with kernel method, cross validation and search of the minimum parameters.
    
    Syntax
    
    (mspeMin, lambdaMin, paramMin) = mspe_crossfold_search(x, w, 
        kernelMethod, nFold, listLambda, tupleParam)
        
    Input
    
    x: (1, nObs), targets
    w: (nDimW, nObs), predictors
    kernelMethod: "Gaussian" or "linear"
    nFold: int, number of sets for crossfold validation 
    listLambda: list of float for lambda parameters
    tupleParam: tuple of list of parameters
    
    Output
    
    mspeMin: float, the minimal mspe
    lambdaMin: the optimal parameter lambda
    paramMin: tuple of optimal parameter
    
    Description
    
    Same as kernel.mspe_crossfold with a search of optimal parameters for lambda an kernel parameters. 
    
    Example
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.rand(100, )
    >>> w = numpy.random.rand(2, 100)
    >>> res = kernel.mspe_crossfold_search(x, w, "Gaussian", 10, [1., 10.], ([1., 10.], ))
    >>> print(res)    

        (0.097227954750672654, 1.0, array([ 10.]))
        
    Example
    
    >>> numpy.random.seed(1)
    >>> x = numpy.random.rand(100, )
    >>> w = numpy.random.rand(2, 100)
    >>> res = kernel.mspe_crossfold_search(x, w, "linear", 10, numpy.arange(1., 100., 10.), ([1], ))
    >>> print(res)    
    
        (0.13769384149525249, 1.0, array([ 1.]))
        
    See also kernel.mspe_crossfold, kernel.mspe
    
    """
    mspeMin = numpy.inf
    allCases = numpy.asarray(util.getListOfCases(tupleParam))
    allCases = allCases.T
    for lambda_ in listLambda: 
        for param in allCases: 
            mspe = mspe_crossfold(x, w, kernelMethod, nFold, lambda_, param)
            if (mspe < mspeMin):
                paramMin = param
                lambdaMin = lambda_
                mspeMin = mspe
    return (mspeMin, lambdaMin, paramMin)
#_______________________________________________________________________________
