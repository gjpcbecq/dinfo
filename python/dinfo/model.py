"""
Some models used in publications and examples 

Contain

GaussianBivariate - generate samples of a Gaussian bivariate model
GaussianChannel - generate samples of a Gaussian channel
GaussianCovariate - generate samples of a Gaussian covariate model
GaussianH - compute the entropy of a Gaussian covariate model
GaussianXY - generate samples of a Gaussian bivariate model
GaussianXYZ - generate samples of a Gaussian trivariate model
GlassMackey - generate samples of a Glass Mackey like system 
aR1Bivariate - generate samples of bivariate AR model of order 1 
aR1Trivariate - generate samples of trivariate AR model of order 1
chain - generate samples of a chained system
chain2 - generate samples of another chained system
coupledLorenzSystems - generate samples of a coupled Lorenz system
fourDimensional - generate samples of a four dimensional system
sameCovariance - generate a matrix with same covariance 

Reference

Amblard, P.-O. and Michel, O., ``Measuring information flow in networks of stochastic processes'', arXiv, 2009, 0911.2873
Amblard, P.-O.; Michel, O. J.; Richard, C. and Honeine, P., ``A Gaussian process regression approach for testing Granger causality between time series data'',  Acoustics, Speech and Signal Processing (ICASSP), 2012 IEEE International Conference on, 3357-3360, 2012.
Amblard, P.-O.; Vincent, R.; Michel, O. J. and Richard, C., ``Kernelizing Geweke's measures of granger causality'', Machine Learning for Signal Processing (MLSP), 2012 IEEE International Workshop on, 1-6, 2012.
Amblard, P.-O. and Michel, O., ``Causal conditioning and instantaneous coupling in causality graphs'', Information Sciences, Elsevier, 2014.
Cover, T. and Thomas, J., ``Elements of information theory'', Wiley Online Library, 1991, 6.
Frenzel, S. and Pompe, B., ``Partial mutual information for coupling analysis of multivariate time series'', Physical review letters,  APS, 99, 204101, 2007.

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
def aR1Bivariate(nObs, cXX=0, cXY=0, cYX=0, cYY=0, gVW=0, sV=1, sW=1, 
    dX=0, dY=0): 
    """Simulate samples from a bivariate aR1 model. 
    
    Syntax
    
    (x, y) = aR1Bivariate(nObs, cXX=1, cXY=0, cYX=1, cYY=0, gVW=0, sV=1, sW=1, dX=0, dY=0)
    
    Input
    
    nObs: int, number of samples
    cXX=0: coef from X to X
    cXY=0: coef from Y to X
    cYX=0: coef from X to Y
    cYY=0: coef from Y to Y
    gVW=0: correlation coefficient between noise V and W
    sV=1: standard deviation noise V 
    sW=1: standard deviation noise V
    dX=0: constant on X
    dY=0: constant on Y
    
    Output 
    
    X: (2, nObs)
    
    Description
    
    This is the example given in P.-O. Amblard and O. Michel, ``Measuring information flow in networks of stochastic processes'', 2009  
    
    Example
    
    >>> numpy.random.seed(1)
    >>> nObs = 1000
    >>> x = model.aR1Bivariate(nObs, cXX=0, cXY=0., cYX=0, cYY=0, gVW=0, sV=1, sW=1, dX=0, dY=0)
    >>> print(corrcoef(x[0, 1:], x[1, :-1]))

    [[ 1.          0.04047559]
     [ 0.04047559  1.        ]]
    
    >>> numpy.random.seed(1)
    >>> nObs = 1000
    >>> x = model.aR1Bivariate(nObs, cXX=0, cXY=0.9, cYX=0, cYY=0, gVW=0, sV=1, sW=1, dX=0, dY=0)
    >>> print(corrcoef(x[0, 1:], x[1, :-1]))  
    
    [[ 1.          0.70234545]
     [ 0.70234545  1.        ]]
 
    """
    GammaW = numpy.array([[1, gVW], [gVW, 1]])
    M = numpy.array([[0.], [0.]])
    W = GaussianCovariate(nObs, M, GammaW)
    iV = 0
    iW = 1
    W[iV, :] *= sV
    W[iW, :] *= sW
    x = numpy.zeros((nObs, ))
    y = numpy.zeros((nObs, ))
    C = numpy.zeros((2, 2))
    iX = 0 
    iY = 1
    C[iX, iX] = cXX
    C[iX, iY] = cXY
    C[iY, iX] = cYX
    C[iY, iY] = cYY
    for t in range(1, nObs):
        tM1 = t - 1
        x[t] = C[iX, iX] * x[tM1] + C[iX, iY] * y[tM1] + dX + W[iX, t]
        y[t] = C[iY, iX] * x[tM1] + C[iY, iY] * y[tM1] + dY + W[iY, t]
    X = numpy.vstack((x, y))
    return X
    
#_______________________________________________________________________________
#_______________________________________________________________________________
def aR1Trivariate(nObs, C, D, S): 
    """Simulate samples from a trivariate aR1 model. 
    
    Syntax
    
    X = aR1Trivariate(nObs, C, D, S)
    
    Input
    
    nObs: int, number of samples
    C: coupling array
    S: covariance matrix of the noise  
    D: constant for the AR
    
    Output 
    
    X: (3, nObs)
    
    Description
    
    This is the example given in P.-O. Amblard and O. Michel, ``Measuring information flow in networks of stochastic processes'', 2009  
    
    Example
    
    >>> numpy.random.seed(1)
    >>> C = numpy.array([[0.4, 0., -0.6], [0.4, 0.5, 0.], [0., 0.5, -0.5]])
    >>> D = numpy.array([0, 0, 0])
    >>> S = numpy.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    >>> X = model.aR1Trivariate(1000, C, D, S)
    >>> print(corrcoef(X[0, 1:], X[1, :-1]))

    [[  1.00000000e+00   2.02002605e-04]
     [  2.02002605e-04   1.00000000e+00]]

    Example
    
    >>> numpy.random.seed(1)
    >>> C = numpy.array([[0.4, 0.5, -0.6], [0.4, 0.5, 0.], [0., 0.5, -0.5]])
    >>> D = numpy.array([0, 0, 0])
    >>> S = numpy.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    >>> X = model.aR1Trivariate(1000, C, D, S)
    >>> print(corrcoef(X))

    [[ 1.          0.64152953]
     [ 0.64152953  1.        ]]

    """
    M = numpy.array([[0.], [0.], [0.]])
    W = GaussianCovariate(nObs, M, S)
    X = numpy.zeros((3, nObs))
    for t in range(nObs-1):
        tM1 = t - 1
        for i in range(3):
            for j in range(3): 
                X[i, t] += C[i, j] * X[j, tM1] 
            X[i, t] += D[i] + W[i, t]
    return X
    
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianXY(nObs, rho): 
    """Simulate samples from Gaussian bivariate model X and Y
    
    Syntax
    
    (x, y) = GaussianXY(nObs, rho)
        
    Input
    
    nObs: float, number of observations
    rho: float, correlation coefficient
    
    Output
    
    x: (1, nObs)
    y: (1, nObs)

    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y) = model.GaussianXY(1000, 0.9) 
    >>> print(corrcoef(x, y))
    
    [[ 1.         0.8934173]
     [ 0.8934173  1.       ]]

    """
    X = numpy.random.randn(2, nObs)
    C = numpy.array([[1, rho], [rho, 1]])
    R = numpy.linalg.cholesky(C)
    Y = numpy.dot(R, X)
    x = Y[0, :] 
    y = Y[1, :]
    return (x, y) 
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianXYZ(nObs, rhoXY, rhoYZ, rhoZX): 
    """Simulate samples from Gaussian trivariate model X, Y and Z
    
    Syntax
    
    (x, y, z) = GaussianXYZ(nObs, rhoXY, rhoYZ, rhoZX)
        
    Input
    
    nObs: float, number of observations
    rhoXY: float, correlation coefficient
    rhoYZ: float, correlation coefficient
    rhoZX: float, correlation coefficient
    
    Output
    
    x: (1, nObs)
    y: (1, nObs)
    z: (1, nObs)

    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(1000, 0.9, 0.8, 0.7) 
    >>> print(corrcoef((x, y, z)))
    
        [[ 1.          0.8934173   0.69159129]
         [ 0.8934173   1.          0.79682835]
         [ 0.69159129  0.79682835  1.        ]]
         
    """
    X = numpy.random.randn(3, nObs)
    C = numpy.array([[1, rhoXY, rhoZX], [rhoXY, 1, rhoYZ],
            [rhoZX, rhoYZ, 1]])
    R = numpy.linalg.cholesky(C)
    Y = numpy.dot(R, X)
    x = Y[0, :]
    y = Y[1, :]
    z = Y[2, :]
    return (x, y, z)
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianH(V): 
    """Compute Gaussian entropy given V matrix of covariance
    
    Syntax
    
    hTh = GaussianH(V)
    
    Input
    
    V: int, float, list, or array
    
    Output
    
    hTh: float
    
    Description
    
    $d$ is the dimension of the variable. 
    $V$ is a coefficient of correlation, or the matrix of covariance. 
    $|V|$ is the determinant of V.
    $$ h = \frac{1}{2} \, \log{(2 \, \pi \, e)^d \, |V|} $$
    
    Example 
    
    >>> V = 1
    >>> print(model.GaussianH(V))
        1.4189385332
        
    >>> V = [[1., 0.8], [0.8, 1]]
    >>> print(model.GaussianH(V)) 
        2.32705144264
        
    """
    V = numpy.asmatrix(V)
    nDim = V.shape[0]
    detV = numpy.linalg.det(V)
    coef = (2. * numpy.pi * numpy.exp(1)) ** nDim
    hTh = 1. / 2. * numpy.log(coef * detV)
    return hTh
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianChannel(nObs, P, N): 
    """Simulate x, y, and z from a Gaussian channel y = x + z. 
    
    Syntax
    
    (x, y, z) = GaussianChannel(nObs, P, N)
    
    Input
    
    nObs: int, number of observations
    P: variance of X 
    N: variance of Z
    
    Output
    
    (x, y, z): tuple
        x: (nObs, ) 
        y: (nObs, ) 
        z: (nObs, ) 
    
    Description

    X, Y and Z are Gaussian variables. 
    Y is the output of the channel.  
    
    $X \\sim N(0, P)$
    $Z \\sim N(0, N)$
    $Y \\sim N(0, P + N)$

    $Y_i = X_i + Z_i$
    
    h(x) = 1 / 2 \\, \\log{(2\\, \\pi \\, mbox{e}) \\, (P)}
    h(z) = 1 / 2 \\, \\log{(2\\, \\pi \\, mbox{e}) \\, (N)}
    h(y) = 1 / 2 \\, \\log{(2\\, \\pi \\, mbox{e}) \\, (P + N)}
    
    P = 2, N = 1: 
    mi(X; Y) = 1 / 2 \\, \\log{1 + P / N} = 0.5493
    
    see Cover and Thomas, p. 261+
    
    Example 
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianChannel(1000, 2, 1)
    >>> from dinfo import dinfo
    >>> miXY = dinfo.mi(x, y, "Kraskov", (3, ))
    >>> print(miXY)    
    
        0.529399686532
    
    """
    x = numpy.sqrt(P) * numpy.random.randn(1, nObs) 
    z = numpy.sqrt(N) * numpy.random.randn(1, nObs)
    y = x + z
    return (x, y, z)
#_______________________________________________________________________________
#_______________________________________________________________________________
def chain(nObs): 
    """ Simulate samples from the example 5.1. of Amblard et al. 2012 Workshop MLSP

    Syntax

    (x, y, z, e, M, M2) = chain(nObs)
    
    Input
    
    nObs: int, number of samples to simulate
    
    Output
    
    x: (nObs, )
    y: (nObs, )
    z: (nObs, )
    e: (3, nObs), the noise
    M: (3, 3), matrix of linear relations
    M2: (3, 3), matrix of squared relations
    
    Description
    
    x -> y -> z
    
    $$
    \\left \\{
    \\begin{array}{l}
    x_t = a \\, x_{t - 1} + \\epsilon_{x, t} \\\\
    y_t = b \\, y_{t - 1} + d_{xy} \\, x_{t - 1} ^2 + \\epsilon_{y, t} \\\\
    z_t = c \\, z_{t - 1} + c_{yz} \\, y_{t - 1} + \\epsilon_{z, t} \\\\
    (a, b, c, d_{xy}, c_{yz}) = (0.2, 0.5, 0.8, 0.8, 0.7)
    \\end{array}
    \\right .
    $$
    Initial parameters are set to zeros.

    Example

    >>> nObs = 100
    >>> numpy.random.seed(1)
    >>> from dinfo import dinfo
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1, 10], )
    >>> nTrial = 10
    >>> dXY = numpy.zeros((nTrial, ))
    >>> dXYKZ = numpy.zeros((nTrial, ))
    >>> for i in range(nTrial): 
    >>>     (x, y, z, e, M, M2) = model.chain(nObs)
    >>>     dXY[i] = dinfo.gcm(x, y, z, 3, "", "dynamic", "Gaussian", 2, listLambda, param)
    >>>     dXYKZ[i] = dinfo.gcm(x, y, z, 3, "conditional", "dynamic", "Gaussian", 2, listLambda, param)
    >>>     print("dXY:{0:5.2f}, dXYKZ:{1:5.2f}".format(dXY[i], dXYKZ[i]))
    >>> (pXY, stepX, thX, xS) = dinfo.binning.prob(dXY, nBin=10)
    >>> mXY = (thX[0, 0:-1] + thX[0, 1:]) / 2.
    >>> (pXYKZ, stepX, thX, xS) = dinfo.binning.prob(dXYKZ, nBin=10)
    >>> mXYKZ = (thX[0, 0:-1] + thX[0, 1:]) / 2.
    >>> plot(mXY, pXY)
    >>> plot(mXYKZ, pXYKZ)

        dXY:-0.04, dXYKZ:-0.03
        dXY: 0.29, dXYKZ: 0.10
        dXY: 0.02, dXYKZ:-0.01
        dXY: 0.00, dXYKZ:-0.06
        dXY: 0.07, dXYKZ:-0.03
        dXY: 0.30, dXYKZ: 0.06
        dXY:-0.07, dXYKZ:-0.10
        dXY: 0.06, dXYKZ:-0.06
        dXY:-0.03, dXYKZ:-0.03
        dXY: 0.22, dXYKZ: 0.04

    """
    (IX, IY, IZ) = (0, 1, 2)
    nDim = 3
    X = numpy.zeros((nDim, nObs + 1))
    (X[IX, 0], X[IY, 0], X[IZ, 0]) = (0, 0, 0) 
    M = numpy.zeros((nDim, nDim))
    M2 = numpy.zeros((nDim, nDim))
    (M[IX, IX], M[IY, IY], M[IZ, IZ], M2[IY, IX], M[IZ, IY]) = (0.2, 0.5, 
        0.8, 0.8, 0.7)
    E = 1 * numpy.random.randn(nDim, nObs + 1)
    for p in range(1, nObs + 1): 
        pm1 = p - 1
        X[IX, p] = M[IX, IX] * X[IX, pm1] + E[IX, p]
        X[IY, p] = (M[IY, IY] * X[IY, pm1] + M2[IY, IX] * 
            (X[IX, pm1] ** 2) + E[IY, p])
        X[IZ, p] = (M[IZ, IZ] * X[IZ, pm1] + M[IZ, IY] * X[IY, pm1] + 
            E[IZ, p])
    return (X[IX, 1:], X[IY, 1:], X[IZ, 1:], E[:, 1:], M, M2)
#_______________________________________________________________________________
#_______________________________________________________________________________
def fourDimensional(nObs): 
    """Simulate samples from the example 5.2. of Amblard et al. 2012 Workshop MLSP

    Syntax
    
    (w, x, y, z, e, M, M2) = fourDimensional(nObs)
    
    Input
    
    nObs: number of samples to simulate
    
    Output
    
    w: (nObs, ) 
    x: (nObs, ) 
    y: (nObs, ) 
    z: (nObs, ) 
    M: (4, 4), matrix of linear relation
    M2: (4, 4), matrix of squared relations
    E: (4, nObs), the noise with multidimensional Gaussian distribution.
    
    Description
    
    Initial parameters are set to zeros.
    Noise are $\\epsilon_{w, t}$, $\\epsilon_{x, t}$, $\\epsilon_{y, t}$, $\\epsilon_{z, t}$ with covariance given by:  
    \\begin{equation*}
    \\Gamma_{\\epsilon} = 
    \\left( 
    \\begin{array}{cccc}
    1 & \\rho_1 & 0 & \\rho_1 \\, \\rho_2 \\\\
    \\rho_1 & 1 & 0 & \\rho_2 \\\\
    0 & 0 & 1 & \\rho_3 \\\\
    \\rho_1 \\, \\rho_2 & \\rho_2 & \\rho_3 & 1
    \\end{array}
    \\right)
    \\end{equation*}
    with
    $(\\rho_1, \\rho_2, \\rho_3) = (0.66, 0.55, 0.48)$
    
    Th multivariate are given by: 
    \\begin{equation*}
    \\left \\{
    \\begin{array}{lclclclclcl}
    w_t & = & 0.2 \\, w_{t - 1} & - & 0.2 \\, x_{t - 1} ^2 & & & + & 0.3 \\, z_{t - 1} & + & \\epsilon_{w, t} \\\\
    x_t & = & & & 0.3 \\, x_{t - 1} & & & + & 0.3 \\, z_{t - 1} ^2 & + & \\epsilon_{x, t} \\\\
    y_t & = & & & 0.8 \\, x_{t - 1} - 0.5 \\, x_{t - 1} ^2 & - & 0.8 \\, y_{t - 1} & & & + & \\epsilon_{y, t} \\\\
    z_t & = & 0.2 \\, w_{t - 1} & & & & & - & 0.4 \\, z_{t - 1} & + & \\epsilon_{z, t}
    \\end{array}
    \\right .
    \\end{equation*}

    Reference
    
    Amblard et al., ``Kernelizing Geweke's measures of Granger Causality'', workshop on machine learning for signal processing, 2012
    
    Example
    
    >>> nObs = 100
    >>> numpy.random.seed(1)
    >>> (w, x, y, z, e, M, M2) = model.fourDimensional(nObs)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.],)
    >>> from dinfo import dinfo
    >>> print(dinfo.gcm(w, x, y, 3, "", "dynamic", "Gaussian", 10, listLambda, param))   
    
        -0.0141044540402
    
    """
    (IW, IX, IY, IZ) = (0, 1, 2, 3)
    nDim = 4
    X = numpy.zeros((nDim, nObs + 1), dtype='float')
    (X[IW, 0], X[IX, 0], X[IY, 0], X[IZ, 0]) = (0, 0, 0, 0) 
    M = numpy.zeros((nDim, nDim), dtype='float')
    M2 = numpy.zeros((nDim, nDim), dtype='float')
    (M[IW, IW], M[IW, IZ], M2[IW, IX]) = (0.2, 0.3, -0.2) 
    (M[IX, IX], M2[IX, IZ]) = (0.3, 0.3)
    (M[IY, IY], M[IY, IX], M2[IY, IX]) = (-0.8, 0.8, -0.5)
    (M[IZ, IZ], M[IZ, IW]) = (-0.4, 0.2)
    # print("M: " + str(M))
    # print("M2: " + str(M2))
    Gamma = numpy.empty((nDim, nDim))
    (rho1, rho2, rho3) = (0.66, 0.55, 0.48)
    Gamma[0, :] = numpy.array([1, rho1, 0, rho1 * rho2])
    Gamma[1, :] = numpy.array([rho1, 1, 0, rho2])
    Gamma[2, :] = numpy.array([0, 0, 1, rho3])
    Gamma[3, :] = numpy.array([rho1 * rho2, rho2, rho3, 1])
    # Cholesky factorization of Gamma for multidimensional Gaussian
    R = numpy.linalg.cholesky(Gamma); 
    E = 1 * numpy.random.randn(nDim, nObs + 1)
    E = R.dot(E) 
    # print('E: ' + str(numpy.corrcoef(E)))
    for p in range(1, nObs + 1):
        pm1 = p - 1
        X[IW, p] = (M[IW, IW] * X[IW, pm1] + M[IW, IZ] * X[IZ, pm1] + 
            M2[IW, IX] * (X[IX, pm1] ** 2) + E[IW, p])
        X[IX, p] = (M[IX, IX] * X[IX, pm1] + M2[IX, IZ] * 
            (X[IZ, pm1] ** 2) + E[IX, p])
        X[IY, p] = (M[IY, IY] * X[IY, pm1] + M[IY, IX] * X[IX, pm1] + 
            M2[IY, IX] * (X[IX, pm1] ** 2) + E[IY, p])
        X[IZ, p] = (M[IZ, IZ] * X[IZ, pm1] + M[IZ, IW] * X[IW, pm1] + 
            E[IZ, p])
    return (X[IW, 1:], X[IX, 1:], X[IY, 1:], X[IZ, 1:], E[:, 1:], M, M2)
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianBivariate(nObs, rho=1):
    """Simulate samples from two Gaussian covariate with a given correlation 
    coefficient rho.
    
    Syntax
    
    x = GaussianBivariate(nObs, rho=1)
    
    Input
    
    nObs: int, number of samples
    rho=1: float, correlation coefficient
    
    Output
    
    x: (2, nObs)
    
    Example
    
    >>> numpy.random.seed(1)
    >>> x = model.GaussianBivariate(100, rho=0.5)
    >>> from dinfo import dinfo
    >>> hX = dinfo.h(x, "Leonenko", (3, ))
    >>> print(hX)
    >>> print(model.GaussianH([[1., 0.5], [0.5, 1]]))
    
        2.55694961569
        2.69403603018
        
    """
    x = numpy.random.randn(2, nObs)
    V = numpy.array([[1, rho], [rho, 1]])
    R = numpy.linalg.cholesky(V); # Cholesky factorization of V for multidim Gaussian
    x = R.dot(x) 
    return x
#_______________________________________________________________________________
#_______________________________________________________________________________
def sameCovariance(nDim, c): 
    """Generate a covariance matrix with the same correlation for all pairs of  
    variables. 
    
    Syntax
    
    C = sameCovariance(nDim, c)
    
    Input
    
    nDim: number of variable
    c: value of the correlation between variables 
    
    Output
    
    C: (nDim, nDim)
    
    Example
    
    >>> c = 0.9
    >>> C = model.sameCovariance(3, c)
    >>> print(C)
    
        [[ 1.   0.9  0.9]
         [ 0.9  1.   0.9]
         [ 0.9  0.9  1. ]]
         
    """
    C = numpy.empty((nDim, nDim))
    for i in range(nDim): 
        for j in range(nDim): 
            if (i == j): 
                C[i, j] = 1
            else: 
                C[i, j] = c    
    return C
#_______________________________________________________________________________
#_______________________________________________________________________________
def GaussianCovariate(nObs, m, C):
    """
    Simulate samples from Gaussian covariates with a given covariance matrix C
    
    Syntax
    
    x = GaussianCovariate(nObs, m, C)
    
    Input
    
    nObs: int, number of observations
    m: (nDim, )
    C: (nDim, nDim)

    Output
    
    x: (nDim, nObs)
    
    Example
    
    >>> numpy.random.seed(1)
    >>> C = numpy.array([[1, 0.9, 0.8], [0.9, 1, 0.5], [0.8, 0.5, 1]])
    >>> m = numpy.array([[0.], [3.], [-10.]])
    >>> x = model.GaussianCovariate(100, m, C)
    >>> from dinfo import dinfo
    >>> hX = dinfo.h(x, "Leonenko", (3, ))
    >>> print(hX)
    >>> print(model.GaussianH(C))

        2.21772524975
        2.3008040969

    """
    nDim = m.shape[0]
    x = numpy.random.randn(nDim, nObs);
    R = numpy.linalg.cholesky(C); # Cholesky factorization of V for multidim Gaussian
    x = R.dot(x) 
    return x
#_______________________________________________________________________________
#_______________________________________________________________________________
def GlassMackey(nObs, epsilonX, epsilonY, alpha):
    """Simulate samples from equations based on a Glass Mackey model
    used in Amblard et al., A Gaussian Process Regression..., 2012 
    
    Syntax
    
    (x, y) = GlassMackey(nObs, epsX, epsY, alpha)
    
    Input
    
    nObs: int, number of observations
    epsX: (nObs, ) noise on X
    epxY: (nObs, ) noise on Y
    alpha: float, parameter of the model
    
    Output
    
    (x, y): 
        x: (nObs, ) 
        y: (nObs, )
    
    Description
    
    The system is given by this equation: 
    $$
    \\left \\{
    \\begin{array}{l}
    x_t = x_{t - 1} - 0.4 \\, \\left( x_{t - 1} - \\frac{2 \\, x_{t - 4}} 
        {1 + x_{t - 4} ^ {10}} \\right) \\, y_{t - 5} +
        0.3 \\, y_{t - 3} + \\epsilon_{x, t} \\\\
    y_t = y_{t - 1} - 0.4 \\, \\left( y_{t - 1} - \\frac{2 \\, y_{t - 2}}
        {1 + y_{t - 2} ^ {10}} \\right) + 
        \\alpha \\, x_{t - 2} + \\epsilon_{y, t}
    \\end{array} 
    \\right .
    $$
    
    Example
    
    >>> nObs = 100
    >>> numpy.random.seed(1)
    >>> epsX = numpy.random.randn(nObs, ) * 0.01
    >>> epsY = numpy.random.randn(nObs, ) * 0.01
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> from dinfo import dinfo
    >>> for alpha in [0, 0.01, 0.1, 0.2]: 
    >>>     (x, y) = model.GlassMackey(nObs, epsX, epsY, alpha)
    >>>     dXY = dinfo.gprcm(x, y, 6, "Gaussian", listLambda, param)
    >>>     dYX = dinfo.gprcm(y, x, 6, "Gaussian", listLambda, param)
    >>>     print("alpha:{0:5.2f}, dXY:{1:5.2f}, dYX:{2:5.2f}".format(alpha, dXY, dYX))

       alpha: 0.00, dXY:-102.23, dYX:-43.26
       alpha: 0.01, dXY:-78.77, dYX:-44.55
       alpha: 0.10, dXY:-83.92, dYX:-41.02
       alpha: 0.20, dXY:-24.96, dYX:-16.54

    """
    x = numpy.ones((nObs + 5, ))
    y = numpy.ones((nObs + 5, ))
    for t in range(5, nObs + 5): 
        (tm1, tm2, tm3, tm4, tm5) = (t - 1, t - 2, t - 3, t - 4, t - 5)
        x[t] = (x[tm1] - 0.4 * (x[tm1] - (2. * x[tm4]) / 
            (1. + x[tm4] ** 10.)) * y[tm5] + 0.3 * y[tm3] + epsilonX[tm5])
        y[t] = (y[tm1] - 0.4 * (y[tm1] - (2. * y[tm2]) / 
            (1. + y[tm2] ** 10.)) + alpha * x[tm2] + epsilonY[tm5])
    return (x[5:], y[5:])
#_______________________________________________________________________________
#_______________________________________________________________________________
def chain2(nObs, epsX, epsY, epsZ, a=1.8): 
    """
    Simulate samples from a chain model used in Amblard et al., A Gaussian 
    Process Regression..., 2012
    
    Syntax
    
    (x, y, z) = chain2(nObs, epsX, epsY, epsZ, a=1.8): 

    Input
    
    nObs: int, number of samples
    epsX: (1, nObs)
    epsY: (1, nObs)
    epsZ: (1, nObs)
    a=1.8: float, parameter
    
    Output

    x: (1, nObs)
    y: (1, nObs)
    z: (1, nObs)
    
    Description
    
    x(0), y(0) and z(0) are set to 0.  
    $$
    \\left \\{
    \begin{array}{l}
    x(n) = 1 - a \\, x(n-1)^2 + epsX(n) \\\\
    y(n) = 0.8 \\, (1 - a \\, y(n-1)^2) + 0.2 \\, (1 - a \\, x(n-1)^2) + epsY(n) \\\\
    z(n) = 0.8 \\, (1 - a \\, z(n-1)^2) + 0.2 \\, (1 - a \\, y(n-1)^2) + epsZ(n)
    \\right . 
    $$
    
    Example
    
    >>> nObs = 100
    >>> numpy.random.seed(1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1, 10], )
    >>> epsX = numpy.random.randn(nObs, ) * 0.0001
    >>> epsY = numpy.random.randn(nObs, ) * 0.0001
    >>> epsZ = numpy.random.randn(nObs, ) * 0.0001
    >>> (x, y, z) = model.chain2(nObs, epsX, epsY, epsZ)
    >>> dXZ = dinfo.gprcm(x, z, 3, "Gaussian", listLambda, param)
    >>> dXZKY = dinfo.gprcmc(x, z, y, 3, "Gaussian", listLambda, param)
    >>> print("dXZ:{0:5.2f}, dXZKY:{1:5.2f}".format(dXZ, dXZKY))
    
        dXZ:-41.61, dXZKY:-94.62
        
    """
    
    x = numpy.zeros((nObs + 1, ))
    y = numpy.zeros((nObs + 1, ))
    z = numpy.zeros((nObs + 1, ))
    for t in range(1, nObs + 1): 
        tm1 = t - 1
        x2 = x[tm1]
        x2 *= x2
        y2 = y[tm1]
        y2 *= y2
        z2 = z[tm1]
        z2 *= z2
        x[t] = 1 - a * x2 + epsX[tm1]
        y[t] = 0.8 * (1 - a * y2) + 0.2 * (1 - a * x2) + epsY[tm1]
        z[t] = 0.8 * (1 - a * z2) + 0.2 * (1 - a * y2) + epsZ[tm1]
    return (x[1:], y[1:], z[1:])       
#_______________________________________________________________________________
#_______________________________________________________________________________
def coupledLorenzSystems(nObs, K, tau, b=8./3., sigma=10., r=28., tauMax=60):
    """Simulate samples from three coupled Lorenz systems used in Frenzel and 
    Pompe 2007
    
    Syntax
    
    (x, y, z) = coupledLorenzSystems(nObs, K, tau, b=8./3., sigma=10., r=28., 
        tauMax=60)
    
    Input
    
    nObs: number of observations
    K: (3, 3), coupling between systems
    tau: (3, 3), delay between systems
    b=8./3.: float
    sigma=10.: float
    r=28.: float
    tauMax=60: float
    
    Output
    
    x: (3, nObs), X1, X2, X3
    y: (3, nObs), Y1, Y2, Y3
    z: (3, nObs), Z1, Z2, Z3
    
    Description
    
    $$
    \\left \\{
    \\begin{array}{l}
    \\dot{X}_i(t) = \\sigma \\, (Y_i(t) - X_i(t)) \\\\
    \\dot{Y}_i(t) = r \\, X_i(t) - Y_i(t) - X_i(t) \\, Z_i(t) + 
        \\sum_{j \\neq i} K_{ij} \\, Y_j^2(t - \\tau_{ij})\\\\
    \\dot{Z}_i(t) = X_i(t) \\, Y_i(t) - b \\, Z_i(t) 
    \\end{array}
    \\right . 
    $$
    Euler method for integration in 2 steps with integration steps of 0.006. 
    Runge-Kutta method like in the article is not used here for simplicity. 
    All data are returned with a sampling period of 0.003. 
    For getting a sampling period of $\Delta t =0.3$ like in the artcile, use 
    (X, Y, Z) = (x[:, ::100], y[:, ::100], z[:, ::100]) 
    Initialisation is realized by taking uniform values in [0, 0.01]    

    Example
    
    >>> nObs = 10000
    >>> K = numpy.zeros((3, 3))
    >>> K[0, 1] = 0.5
    >>> K[1, 2] = 0.5
    >>> tau = numpy.zeros((3, 3))
    >>> tau[0, 1] = 1000
    >>> tau[1, 2] = 1500
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.coupledLorenzSystems(nObs, K, tau)
    >>> (X, Y, Z) = (x[:, ::100], y[:, ::100], z[:, ::100])
    >>> from dinfo import dinfo
    >>> tau = 10
    >>> Y1 = Y[0, tau:]
    >>> Y2 = Y[1, :-tau]
    >>> miY1Y2 = dinfo.mi(Y1, Y2, 'Frenzel', (3, 'Euclidean'))
    >>> print("tau:{0:3.0f}, miY1Y2:{1:5.2f}".format(tau, miY1Y2))

        tau: 10, miY1Y2: 0.37

    """
    x = 0.01 * numpy.random.rand(3, nObs + tauMax)
    y = 0.01 * numpy.random.rand(3, nObs + tauMax)
    z = 0.01 * numpy.random.rand(3, nObs + tauMax)
    for t in range(tauMax, nObs + tauMax):
        tm1 = t - 1
        for i in range(3):
            sK = 0
            for j in range(3): 
                yd = y[j, t - tau[i, j]]
                sK += K[i, j] * yd * yd
            # TODO implements Runge Junta method
            # here Euler with two steps (simpler)
            xp = sigma * (y[i, tm1] - x[i, tm1]) 
            yp = x[i, tm1] * (r - z[i, tm1]) - y[i, tm1] + sK
            zp = x[i, tm1] * y[i, tm1] - b * z[i, tm1]
            xt = x[i, tm1] + 0.006 * xp
            yt = y[i, tm1] + 0.006 * yp 
            zt = z[i, tm1] + 0.006 * zp
            xp = sigma * (yt - xt) 
            yp = xt * (r - zt) - yt + sK
            zp = xt * yt - b * zt
            x[i, t] = xt + 0.006 * xp
            y[i, t] = yt + 0.006 * yp 
            z[i, t] = zt + 0.006 * zp
    s1 = slice(tauMax, nObs + tauMax)
    return (x[:, s1], y[:, s1], z[:, s1])
#_______________________________________________________________________________
  
        
            
        
    

