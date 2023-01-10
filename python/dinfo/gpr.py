"""
Compute causal measures based on Gaussian processes  

Contain

gprcm_Gaussian
gprcm_linear
gprcmc_Gaussian
gprcmc_linear
logP_Gaussian
logP_linear

Reference

Amblard, P.-O.; Michel, O. J.; Richard, C. and Honeine, P., ``A Gaussian process regression approach for testing Granger causality between time series data'', Acoustics, Speech and Signal Processing (ICASSP), 2012 IEEE International Conference on, 2012, 3357-336.

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
import kernel
import util

#_______________________________________________________________________________
def logP_Gaussian(xL, xP, xT, sigma, beta):
    """Compute log-evidence for Gaussian process regression with Gaussian kernel
    
    Syntax
    
    (logP, mT, vT, VT) = logP_Gaussian(xL, xP, xT, sigma, beta)
    
    Input
    
    xL: (nObsL, ) or (nDimXL, nObsL), learning set
    xP: (nObsL, ) or (nDimXP, nObsL), prediction or target values
    xT: (nObsT, ) or (nDimXL, nObsT), test set
    sigma: float, hyperparameter of the Gaussian process
    beta: float, parameter of the Gaussian kernel
    
    Output
    
    logP: log evidence
    mT: predictive mean for test value
    vT: predictive covariance matrix for test value 
    VT: a posteriori covariance matrix
    
    Description
    
    $$ logP = a + b + c $$ 
    with 
    $$ a = - 1 / 2 \\, xP^t \\, (K^2 + \\sigma \\, I)^{-1} \\, xP 
    $$ b = - \\sum \\log{(diag(K^2 + \\sigma \\, I)} $$
    $$ c = - 1 / 2 \\, n \\, \\log{(2.0 \\, \\pi)} $$
    
    Example 
    
    >>> xL = numpy.array([1., 2., 3.]) 
    >>> xP = numpy.array([2., 3., 4.]) 
    >>> xT = numpy.array([0., 2., 3., 5.])
    >>> sigma = 0.01 
    >>> beta = 10
    >>> (logP, mT, vT, VT) = gpr.logP_Gaussian(xL, xP, xT, sigma, beta)
    >>> print((logP, mT, vT, VT))
    
        (array([[-28.75677914]]), array([[ 1.02076243],
               [ 3.00837434],
               [ 3.99307497],
               [ 5.65598096]]), array([[ 0.99000033,  0.99000033,  0.9607414 ,  0.85210118],
               [-0.13655324,  0.14071189,  0.27517886,  0.49746301],
               [ 0.01015415,  0.00529091,  0.03405794,  0.14273167]]), array([[  1.14944294e-03,  -1.50284029e-04,   2.76100236e-05,
                  1.70119150e-03],
               [ -1.50284029e-04,   7.15062162e-05,   1.43864643e-05,
                 -4.03415648e-04],
               [  2.76100236e-05,   1.43864643e-05,   9.26065558e-05,
                  3.88100095e-04],
               [  1.70119150e-03,  -4.03415648e-04,   3.88100095e-04,
                  6.08179654e-03]]))
                  
    """
    if (xL.ndim == 1):
        xL = xL.reshape(1, xL.size)
    if (xP.ndim == 1):
        xP = xP.reshape(1, xP.size)
    if (xT.ndim == 1):
        xT = xT.reshape(1, xT.size)
    VERBOSE = False
    solve = numpy.linalg.solve
    # xP = f
    n = xL.shape[1]
    if VERBOSE: print("n: "+ str(n))
    s2 = sigma ** 2.0
    if VERBOSE: print("xL: " + str(xL))
    if VERBOSE: print("beta: " + str(beta))
    G_LL = kernel.GaussianGram(xL, beta)
    if VERBOSE: print("G_LL: \n " + str(G_LL))
    I = numpy.identity(n)
    if VERBOSE: print("I: " + str(I))
    M = G_LL + s2 * I
    # f = xP
    # f^t \, M^{-1} \, f 
    # fx = M^{-1} \, f = (L \, L^t)^{-1} \, f 
    # fx = (L^t)^{-1} \, L^{-1} \, f 
    # fx = L^t \ (L \ f)
    # invMDotXP = fx
    L = numpy.linalg.cholesky(M)
    if VERBOSE: print("inv M \n " + str(numpy.linalg.inv(M)))
    if VERBOSE: 
        print("M \n " + str(M))
        print("inv M \n " + str(numpy.linalg.inv(M)))
        print("xP: " + str(xP))
        print(numpy.linalg.inv(M).shape)
        print(xP.T.shape)
    # invMDotXP = solve(L.T, solve(L, xP))
    invMDotXP = numpy.dot(numpy.linalg.inv(M), xP.T)
    if VERBOSE: print("fx: inv M dot Xp" + str(invMDotXP))
    a = - (1. / 2.) * numpy.dot(xP, invMDotXP)
    b = - numpy.sum(numpy.log(numpy.diag(L)))
    '''
    detM = numpy.linalg.det(M)
    if (detM < 0.001): 
        detM = 0.001
    b = - (1. / 2.) * numpy.log(detM)
    '''
    c = - (1. / 2.) * n * numpy.log(2.0 * numpy.pi) 
    logP = a + b # + c
    if VERBOSE: print("a: {0}, b: {1}, c: {2}, d: {3}".format(a, b, c, logP))
    (G_TL, G_LT)= kernel.GaussianGramXY(xT, xL, beta)
    G_TT = kernel.GaussianGram(xT, beta)
    if VERBOSE: print("G_LT: " + str(numpy.floor(G_LT * 100) / 100))
    mT = numpy.dot(G_TL, invMDotXP)
    vT = solve(L, G_LT)
    VT = G_TT - numpy.dot(vT.T, vT)
    """
    # use Reduced-rank Approximations of the Gram Matrix
    # (M^t + s2 \, I_n)^{-1} =
    # = s2 \, I_n - Q \, (s2 \, I_q + Q^t \, Q)^{-1} \, Q^t 
    # we only have to inverse a q x q matrix now. 
    [u, s, d] = numpy.linalg.svd(M)
    if VERBOSE: print(d[:, :3].T)
    Q = numpy.dot(numpy.dot(u[:, :3], numpy.diag(s[:3])), d[:, :3].T)
    N = numpy.dot(s2, numpy.identity(n)) + numpy.dot(Q.T, Q)
    if VERBOSE: print("N: \n" + str(N) + str(type(N)) + str(N.dtype))
    Ninv = numpy.linalg.inv(N)
    Minv = numpy.dot(s2, numpy.identity(n)) - numpy.dot(numpy.dot(Q, Ninv), 
        Q.T)
    if VERBOSE: print("Minv Approx: \n" + str(Minv))
    """
    return (logP, mT, vT, VT)
#_______________________________________________________________________________
#_______________________________________________________________________________
def gprcm_Gaussian(x, y, p, listSigma=[1.], listBeta=[1.]): 
    """Compute Gaussian process regression causality measure with Gaussian kernel
    
    Syntax
    
    (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = 
        gprcm_Gaussian(x, y, p, listSigma=[1.], listBeta=[1.])
    
    Input
    
    x: (nObs, ) or (nDimX, nObs)
    y: (nObs, ) or (nDimY, nObs)
    p: int, order of the model 
    listSigma: list of sigma values to evaluate with len nSigma 
    listBeta: list of beta values to evaluate with len nBeta
    
    Output
    
    dXY: float, max(logP2) - max(logP1) = max(log(P(fy | x, y))) - max(log(P(fy | y)))
    resP2: (nSigma, nBeta), result of logP2 for parameters
    resP1: (nSigma, nBeta), result of logP1 for parameters
    optimalParamP2: tuple, (beta2Optimal, sigma2Optimal)
    optimalParamP1: tuple, (beta1Optimal, sigma1Optimal)
    
    Example
    
    >>> x = numpy.array([[1., 2., 3., 4., 5., 6., 7.]]) 
    >>> y = numpy.array([[2., 3., 4., 5., 6., 7., 8.]]) 
    >>> listSigma = [0.01, 0.1, 1.]
    >>> listBeta = [0.1, 1, 10]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = gpr.gprcm_Gaussian(x, y, 3, listSigma, listBeta)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))

    (array([[-0.80368898]]), array([[-90.66725499, -90.35132907, -41.94985818],
       [-89.83426866, -89.52449333, -35.96341527],
       [-48.56204849, -48.482889  , -25.27876475]]), array([[-90.66725499, -84.77962603, -43.32683286],
       [-89.83426866, -84.05714401, -34.99694999],
       [-48.56204849, -47.03115009, -24.47507576]]), (1.0, 10), (1.0, 10))     

    Example
    
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listSigma = [0.01, 0.1, 1.]
    >>> listBeta = [0.1, 1, 10]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) =  gpr.gprcm_Gaussian(x, y, 2, listSigma, listBeta)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))

    (array([[ 1.54176457]]), array([[ -1.33077579e+02,  -1.84303864e+03,  -3.52127534e+05],
       [ -1.33136104e+02,  -6.09274374e+02,  -3.83647778e+03],
       [ -1.45477491e+02,  -1.44927764e+02,  -1.35801243e+02]]), array([[ -1.37138307e+02,  -1.69373891e+05,  -3.84057051e+05],
       [ -1.34619344e+02,  -2.47271908e+03,  -3.97103904e+03],
       [ -1.44626913e+02,  -1.40463311e+02,  -1.35453805e+02]]), (0.01, 0.1), (0.1, 0.1))
       
    """
    VERBOSE = False
    if (x.ndim == 1):
        if VERBOSE: print("xndim = 1")
        x = x.reshape((1, x.size))
    if (y.ndim == 1): 
        if VERBOSE: print("yndim = 1")
        y = y.reshape((1, y.size))
    if VERBOSE: 
        print("_______________________\n p={0}".format(p))
        print("x\n{0}".format(x))
        print("y\n{0}".format(y))
    yTM1 = util.getTM1(y, p)
    z = numpy.vstack((x, y))
    zTM1 = util.getTM1(z, p)
    fy = y
    if VERBOSE: print("fy: \n {0}".format(fy))
    # logP2
    optimalParamP2 = (numpy.nan, numpy.nan)
    resP2 = numpy.empty((len(listSigma), len(listBeta)))
    solve = numpy.linalg.solve
    maxLogP2 = -numpy.inf 
    xL = zTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    if VERBOSE: 
        print("P2 (XY)")
        print("xL: \n {0}".format(xL))
        print("xP: \n {0}".format(xP))
        print("xT: \n {0}".format(xT))
    for (iSigma, sigma) in enumerate(listSigma): 
        for (iBeta, beta) in enumerate(listBeta): 
            (logP2, mT, vT, VT) = logP_Gaussian(xL, xP, xT, sigma, beta)
            resP2[iSigma, iBeta] = logP2[0, 0]
            if (logP2[0, 0] > maxLogP2): 
                maxLogP2 = logP2[0, 0]
                optimalParamP2 = (sigma, beta)
            
    # logP1
    optimalParamP1 = (numpy.nan, numpy.nan)
    resP1 = numpy.empty((len(listSigma), len(listBeta)))
    maxLogP1 = -numpy.inf
    xL = yTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    if VERBOSE: 
        print("P1 (X)")
        print("xL: \n {0}".format(xL))
        print("xP: \n {0}".format(xP))
        print("xT: \n {0}".format(xT))
    for (iSigma, sigma) in enumerate(listSigma): 
        for (iBeta, beta) in enumerate(listBeta): 
            (logP1, mT, vT, VT) = logP_Gaussian(xL, xP, xT, sigma, beta)
            resP1[iSigma, iBeta] = logP1[0, 0]
            if (logP1[0, 0] > maxLogP1): 
                maxLogP1 = logP1[0, 0]
                optimalParamP1 = (sigma, beta)
    if VERBOSE: 
        print("lXY: {0}".format(maxLogP2))
        print("lX: {0}".format(maxLogP1))
    dXY = maxLogP2 - maxLogP1
    return (dXY, resP2, resP1, optimalParamP2, optimalParamP1)
#_______________________________________________________________________________
#_______________________________________________________________________________
def gprcmc_Gaussian(x, y, z, p, listSigma=[1.], listBeta=[1.]): 
    """Compute Gaussian process regression causality measure conditionally to z with Gaussian kernel. 
    
    Syntax
    
    (dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1) = 
        gprcmc_Gaussian(x, y, z, p, listSigma=[1.], listBeta=[1.])
    
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    z: (nDimZ, nObs)
    p: order of the model 
    listSigma: list of sigma values to evaluate with len nSigma 
    listBeta: list of beta values to evaluate with len nBeta 
    
    Output
    
    dXYkZ: float = max(logP2) - max(logP1) = max(log(P(fy | x, y, z))) - max(log(P(fy | y, z)))
    logP2: (nSigma, nBeta) result of logP2 for parameters
    logP1: (nSigma, nBeta) result of logP1 for parameters
    optimalParamP2: tuple, optimal (sigma, lambda) value for P2 
    optimalParamP1: tuple, optimal (sigma, lambda) value for P1
    
    Description
    
    Influence of x on y conditionally to z : x -> y | z  
    If x causes y, dXYkZ = 0 
    
    Example 
    
    >>> x = numpy.array([[1., 2., 3., 4., 5., 6., 7.]]) 
    >>> y = numpy.array([[2., 3., 4., 5., 6., 7., 8.]]) 
    >>> z = numpy.array([[1., 1., 1., 1., 1., 1., 1.]]) 
    >>> listSigma = [0.01, 0.1, 1.]
    >>> listBeta = [0.1, 1, 10]
    >>> (dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1) = gpr.gprcmc_Gaussian(x, y, z, 3, listBeta, listSigma)
    >>> print((dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1))

    (array([[-0.80368898]]), array([[-90.66725499, -90.35132907, -41.94985818],
       [-89.83426866, -89.52449333, -35.96341527],
       [-48.56204849, -48.482889  , -25.27876475]]), array([[-90.66725499, -84.77962603, -43.32683286],
       [-89.83426866, -84.05714401, -34.99694999],
       [-48.56204849, -47.03115009, -24.47507576]]), (1.0, 10), (1.0, 10))
       
    Example
    
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listSigma = [0.01, 0.1, 1.]
    >>> listBeta = [0.1, 1, 10.]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) =  gpr.gprcmc_Gaussian(x, y, z, 2, listSigma, listBeta)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))
    
    (array([[ 0.00110641]]), array([[ -1.32904146e+02,  -1.64256233e+02,  -2.81336443e+05],
       [ -1.32966864e+02,  -1.58936565e+02,  -3.64483634e+03],
       [ -1.45443965e+02,  -1.44073426e+02,  -1.35980368e+02]]), array([[ -1.32905253e+02,  -6.45229667e+02,  -3.35174829e+05],
       [ -1.32967949e+02,  -2.89346413e+02,  -3.74010523e+03],
       [ -1.45444242e+02,  -1.42470307e+02,  -1.35561437e+02]]), (0.01, 0.1), (0.01, 0.1))

       
    """
    VERBOSE = False
    if (x.ndim == 1):
        if VERBOSE: print("xndim = 1")
        x = x.reshape((1, x.size))
    if (y.ndim == 1): 
        if VERBOSE: print("yndim = 1")
        y = y.reshape((1, y.size))
    if (z.ndim == 1): 
        if VERBOSE: print("yndim = 1")
        z = z.reshape((1, z.size))
    if VERBOSE: 
        print("_______________________\n p={0}".format(p))
        print("x\n{0}".format(x))
        print("y\n{0}".format(y))
        print("z\n{0}".format(z))
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    zTM1 = util.getTM1(z, p)
    xyz = numpy.vstack((x, y, z))
    xyzTM1 = util.getTM1(xyz, p)
    yz = numpy.vstack((y, z))
    yzTM1 = util.getTM1(yz, p)
    fy = y
    if VERBOSE: print("fy: \n {0}".format(fy))
    solve = numpy.linalg.solve
    # logP2
    optimalParamP2 = (numpy.nan, numpy.nan)
    resP2 = numpy.empty((len(listSigma), len(listBeta)))
    maxLogP2 = -numpy.inf 
    xL = xyzTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    if VERBOSE: 
        print("P2 (XY)")
        print("xL: \n {0}".format(xL))
        print("xP: \n {0}".format(xP))
        print("xT: \n {0}".format(xT))
    for (iSigma, sigma) in enumerate(listSigma): 
        for (iBeta, beta) in enumerate(listBeta): 
            (logP2, mT, vT, VT) = logP_Gaussian(xL, xP, xT, sigma, beta)
            resP2[iSigma, iBeta] = logP2[0, 0]
            if (logP2 > maxLogP2): 
                maxLogP2 = logP2[0, 0]
                optimalParamP2 = (sigma, beta)
    # logP1
    optimalParamP1 = (numpy.nan, numpy.nan)
    resP1 = numpy.empty((len(listSigma), len(listBeta)))
    maxLogP1 = -numpy.inf
    xL = yzTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    if VERBOSE: 
        print("P1 (X)")
        print("xL: \n {0}".format(xL))
        print("xP: \n {0}".format(xP))
        print("xT: \n {0}".format(xT))
        print("listBeta: {0} \n".format(listBeta))
        print("listSigma: {0} \n".format(listSigma))
    for (iSigma, sigma) in enumerate(listSigma): 
        for (iBeta, beta) in enumerate(listBeta): 
            (logP1, mT, vT, VT) = logP_Gaussian(xL, xP, xT, sigma, beta)
            resP1[iSigma, iBeta] = logP1[0, 0]
            if (logP1 > maxLogP1): 
                maxLogP1 = logP1[0, 0]
                optimalParamP1 = (sigma, beta)
    if VERBOSE: 
        print("lXY: {0}".format(maxLogP2))
        print("lX: {0}".format(maxLogP1))
        print("logP2: {0} \n".format(resP2))
        print("logP1: {0} \n".format(resP1))
    dXYkZ = maxLogP2 - maxLogP1
    return (dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1)
#_______________________________________________________________________________
#_______________________________________________________________________________
# Linear kernels
#_______________________________________________________________________________
#_______________________________________________________________________________
def logP_linear(xL, xP, xT, sigma):
    """Compute log-evidence for Gaussian process regression with linear kernel
    
    Syntax
    
    (logP, mT, vT, VT) = logP_linear(xL, xP, xT, sigma)
    
    Input
    
    xL: (nObs, ) or (nDimL, nObsL), learning set
    xP: (nObs, ) or (nDimP, nObsL), prediction or target values
    xT: (nDimL, nObsT), test set
    sigma: float, hyperparameter of the Gaussian process
    
    Output
    
    logP: log evidence
    mT: predictive mean for test value
    vT: predictive covariance matrix for test value 
    VT: a posteriori covariance matrix
    
    Description
    
    $$ logP = a + b + c $$ 
    with 
    $$ a = - 1 / 2 \\, xP^t \\, (K^2 + sigma \\, I)^{-1} \\, xP $$
    $$ b = - \\sum \\log{(diag(K^2 + \\sigma \\, I)} $$
    $$ c = - 1 / 2 \\, n \\, \\log{(2.0 \\, \\pi)} $$
       
    Example 
    
    >>> xL = numpy.array([1., 2., 3.]) 
    >>> xP = numpy.array([2., 3., 4.]) 
    >>> xT = numpy.array([0., 2., 3., 5.])
    >>> listSigma = 0.1 
    >>> (logP, mT, vT, VT) = gpr.logP_linear(xL, xP, xT, listSigma)
    >>> print((logP, mT, vT, VT))
    
    (array([[-21.91978234]]), array([[ 0.        ],
       [ 2.8551035 ],
       [ 4.28265525],
       [ 7.13775874]]), array([[ 0.        ,  1.99007438,  2.98511157,  4.97518595],
       [-0.        ,  0.17781993,  0.2667299 ,  0.44454983],
       [-0.        ,  0.07161654,  0.10742481,  0.17904134]]), array([[ 0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.0028551 ,  0.00428266,  0.00713776],
       [ 0.        ,  0.00428266,  0.00642398,  0.01070664],
       [ 0.        ,  0.00713776,  0.01070664,  0.0178444 ]])) 
    
    """
    if (xL.ndim == 1):
        xL = xL.reshape(1, xL.size)
    if (xP.ndim == 1):
        xP = xP.reshape(1, xP.size)
    if (xT.ndim == 1):
        xT = xT.reshape(1, xT.size)
    
    solve = numpy.linalg.solve
    # xP = f
    n = xL.shape[1]
    s2 = sigma ** 2.0
    G_LL = kernel.LinearGram(xL)
    I = numpy.asarray(numpy.identity(n))
    M = G_LL + s2 * I
    # f = xP
    # f^t \, M^{-1} \, f 
    # fx = M^{-1} \, f = (L \, L^t)^{-1} \, f 
    # fx = (L^t)^{-1} \, L^{-1} \, f 
    # fx = L^t \ (L \ f)
    # invMDotXP = fx
    L = numpy.linalg.cholesky(M)
    # invMDotXP = solve(L.T, solve(L, xP))
    invMDotXP = numpy.dot(numpy.linalg.inv(M), xP.T)
    a = - (1. / 2.) * numpy.dot(xP, invMDotXP)
    b = - numpy.sum(numpy.log(numpy.diag(L)))
    c = - (1. / 2.) * n * numpy.log(2.0 * numpy.pi) 
    logP = a + b + c
    (G_TL, G_LT)= kernel.LinearGramXY(xT, xL)
    G_TT = kernel.LinearGram(xT)
    mT = numpy.dot(G_TL, invMDotXP)
    vT = solve(L, G_LT)
    VT = G_TT - numpy.dot(vT.T, vT)
    return (logP, mT, vT, VT)
#_______________________________________________________________________________
#_______________________________________________________________________________
def gprcm_linear(x, y, p, listSigma=[1.]): 
    """ Gaussian process regression causality measure with linear kernel 
    
    Syntax
    
    (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = 
        gprcm_linear(x, y, p, listSigma=[1.]) 
        
    Input 
    
    x: (nObs, ) or (nDimX, nObs)
    y: (nObs, ) or(nDimY, nObs)
    p: int, order of the model 
    listSigma: list of sigma values to evaluate 
    
    Output 
    
    dXY: float = maxlog(P2) - maxlog(P1) = max(log(P(fy | x, y))) - max(log(P(fy | y)))
    resP2: (nSigma, ) result of logP2 for parameters
    resP1: (nSigma, ) result of logP1 for parameters
    optimalParamP2: tuple, optimal value for P2 
    optimalParamP1: tuple, optimal value for P1
    
    Example
    
    >>> x = numpy.array([1., 2., 3., 4., 5., 6., 7.]) 
    >>> y = numpy.array([2., 3., 4., 5., 6., 7., 8.]) 
    >>> listSigma = [0.01, 0.1, 1.]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = gpr.gprcm_linear(x, y, 3, listSigma)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))
    
        (array([[-0.33730095]]), array([ 1.636948  , -2.96854499, -7.64720669]), array([ 1.97424895, -2.62375072, -7.24152979]), 0.01, 0.01)
        
    Example
    
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listSigma = [0.01, 0.1, 1., 10.]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = gpr.gprcm_linear(x, y, 3, listSigma)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))
    
        (array([[-2.82576061]]), array([ -3.91702106e+05,  -3.81060591e+03,  -1.39305032e+02,
        -3.14407459e+02]), array([ -4.06022493e+05,  -3.94297876e+03,  -1.36479272e+02,
        -3.13828597e+02]), (1.0,), (1.0,))
    
    """
    if (x.ndim == 1): 
        x = x.reshape((1, x.size))
    if (y.ndim == 1): 
        y = y.reshape((1, y.size))
    yTM1 = util.getTM1(y, p)
    z = numpy.vstack((x, y))
    zTM1 = util.getTM1(z, p)
    fy = y
    # logP2
    optimalParamP2 = (numpy.nan)
    resP2 = numpy.empty((len(listSigma)))
    solve = numpy.linalg.solve
    maxLogP2 = - numpy.inf 
    xL = zTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    for (iSigma, sigma) in enumerate(listSigma): 
        (logP2, mT, vT, VT) = logP_linear(xL, xP, xT, sigma)
        resP2[iSigma] = logP2
        if (logP2 > maxLogP2): 
            maxLogP2 = logP2
            optimalParamP2 = (sigma, )
    # logP1
    optimalParamP1 = (numpy.nan)
    resP1 = numpy.empty((len(listSigma)))
    maxLogP1 = -numpy.inf
    xL = yTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    for (iSigma, sigma) in enumerate(listSigma): 
        (logP1, mT, vT, VT) = logP_linear(xL, xP, xT, sigma)
        resP1[iSigma] = logP1
        if (logP1 > maxLogP1): 
            maxLogP1 = logP1
            optimalParamP1 = (sigma, )
    dXY = maxLogP2 - maxLogP1
    return (dXY, resP2, resP1, optimalParamP2, optimalParamP1)
#_______________________________________________________________________________
#_______________________________________________________________________________
def gprcmc_linear(x, y, z, p, listSigma=[1.]): 
    """Compute Gaussian process regression causality measure conditionally to z with linear kernel
    
    Syntax 
    
    (dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1) = 
        gprcmc_linear(x, y, z, p, listSigma=[1.])
    
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    z: (nDimZ, nObs)
    p: int, order of the model 
    listSigma: list of sigma values to evaluate with len nSigma 
    
    Output
    
    dXYkZ = max(logP2) - max(logP1) = max(log(P(fy | x, y, z))) - max(log(P(fy | y, z)))
    resP2 has shape (len(listSigma)) result of logP2 for parameters
    resP1 has shape (len(listSigma)) result of logP1 for parameters
    optimalParamP2 optimal sigma value for P2 
    optimalParamP1 optimal sigma value for P1

    Description
    
    Influence of x on y conditionally to z : x -> y | z  
    If x causes y, dXYkZ = 0 
    
    Example 

    >>> x = numpy.array([[1., 2., 3., 4., 5., 6., 7.]]) 
    >>> y = numpy.array([[2., 3., 4., 5., 6., 7., 8.]]) 
    >>> z = numpy.array([[1., 1., 1., 1., 1., 1., 1.]]) 
    >>> listSigma = [0.01, 0.1, 1.]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = gpr.gprcmc_linear(x, y, z, 3, listSigma)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))
    
    (array([[-0.4961893]]), array([ 1.61981621, -2.98641166, -7.69428487]), array([ 2.11600551, -2.4914243 , -7.30027411]), 0.01, 0.01)

    Example
    
    >>> from dinfo import model
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listSigma = [0.01, 0.1, 1., 10.]
    >>> (dXY, resP2, resP1, optimalParamP2, optimalParamP1) = gpr.gprcmc_linear(x, y, z, 3, listSigma)
    >>> print((dXY, resP2, resP1, optimalParamP2, optimalParamP1))
    
    (array([[-2.06718144]]), array([ -3.86781768e+05,  -3.77166547e+03,  -1.43220949e+02,
        -3.15315592e+02]), array([ -3.89277832e+05,  -3.78855892e+03,  -1.41153768e+02,
        -3.14742982e+02]), 1.0, 1.0)
    
    """
    if (x.ndim == 1):
        x = x.reshape((1, x.size))
    if (y.ndim == 1): 
        y = y.reshape((1, y.size))
    if (z.ndim == 1): 
        z = z.reshape((1, z.size))
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    zTM1 = util.getTM1(z, p)
    xyz = numpy.vstack((x, y, z))
    xyzTM1 = util.getTM1(xyz, p)
    yz = numpy.vstack((y, z))
    yzTM1 = util.getTM1(yz, p)
    fy = y
    # logP2
    optimalParamP2 = (numpy.nan)
    resP2 = numpy.empty((len(listSigma)))
    maxLogP2 = -numpy.inf 
    xL = xyzTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    for (iSigma, sigma) in enumerate(listSigma): 
        (logP2, mT, vT, VT) = logP_linear(xL, xP, xT, sigma)
        resP2[iSigma] = logP2
        if (logP2 > maxLogP2): 
            maxLogP2 = logP2
            optimalParamP2 = (sigma)
    # logP1
    optimalParamP1 = (numpy.nan)
    resP1 = numpy.empty((len(listSigma)))
    maxLogP1 = -numpy.inf
    xL = yzTM1[:, p:]
    xP = fy[:, p:]
    xT = xL
    for (iSigma, sigma) in enumerate(listSigma):
        (logP1, mT, vT, VT) = logP_linear(xL, xP, xT, sigma)
        resP1[iSigma] = logP1
        if (logP1 > maxLogP1): 
            maxLogP1 = logP1
            optimalParamP1 = (sigma)
    dXYkZ = maxLogP2 - maxLogP1
    return (dXYkZ, resP2, resP1, optimalParamP2, optimalParamP1)
#_______________________________________________________________________________
    
