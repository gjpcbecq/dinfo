"""
Directed information and causality measures. 

For entropy estimation and mutual information, only the continuous case is treated using differential information. 

Contain

h - entropy
hc - conditional entropy
mi - mutual information
mic - conditional or partial mutual information

te - transfert entropy
tec - conditional transfert entropy
iie - instantaneous information exchange
iieu - instantaneous unconditional information exchange
iiec - instantaneous conditional information exchange

gcm - Geweke's causal measure wrapper to gcmd, gcmi, gcmcd, gcmci
gcmd - Geweke's dynamic causal measure
gcmi - Geweke's instantaneous causal measure
gcmcd - Geweke's conditional dynamic causal measure
gcmci - Geweke's conditional instantaneous causal measure

gprcm - Gaussian process regression causal measure
gprcmc - Gaussian process regression causal measure conditional 

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
import util
import binning
import funFrenzel
import funGewekeKernel
import funKraskov
import numpy
import gpr
#_______________________________________________________________________________
def h(x, method="bin", param=(2, )):
    """Compute differential entropy h(X)
    
    Syntax
    
    hX = h(x, method="bin", param=(2, ))

    Input
    
    x: (nDim, nObs)
    method="bin": {"bin", "binning"}, {"Kraskov", "Leonenko"}
    param=(2, ): tuple of parameters
        
    Output
    
    hX: float

    Description
    
    $$ h(x) $$
    This is the differential entropy estimator. It is a wrapper to binning.h, funKraskov.h depending on the method used. 
    
    Example
    
    >>> numpy.random.seed(1)
    >>> x = 2. * numpy.random.rand(3, 1000)
    >>> hX = dinfo.h(x)
    >>> print("h: " + hX)
    >>> hX = dinfo.h(x, (2, ))
    >>> print("h: " + hX)
    >>> hX = dinfo.h(x, "Kraskov", (10, ))
    >>> print("h: " + str(hX))
    >>> hTh = 3 * log(2)
    >>> print('h theory: ' + str(hTh))
    
        h: 2.0669628057
        h: 1.45830014457
        h: 2.24536219883
        h theory: 2.07944154168

    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.h(x))
    >>> print(dinfo.h(x, method="Kraskov", param=(10, )))
    
        1.49800633435
        1.26651766507

    See also binning.h, funKraskov.h 
    
    """
    if (method in {"bin", "binning"}): 
        h = binning.h
    elif (method in {"Kraskov", "Leonenko"}):
        h = funKraskov.h
    else : 
        print("unknown method... check argument")
        h = lambda a, b: numpy.nan
    hX = h(x, param[0])
    return hX
#_______________________________________________________________________________
#_______________________________________________________________________________
def hc(x, y, method="bin", param=(2, )):
    """ Compute conditional entropy of x given y, h(x | y)
    
    Syntax
    
    hXkY = hc(x, y, method="bin", param=(2, ))
        
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    method="bin": {"bin", "binning"}, {"Kraskov", "Leonenko"}
    param=(2, ): tuple of parameters
    
    Output
    
    hXkY: float

    Description
    
    $$ h(x | y) $$
    wrapper to binning.hc, funKraskov.hc
    
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.hc(x, y))
    >>> print(dinfo.hc(x, y, method="Leonenko", param=(10, )))
    
        1.29507898934
        0.522505196932
    
    See also binning.hc, funKraskov.hc
    
    """
    if (method in {"bin", "binning"}): 
        hc = binning.hc
    elif (method in {"Kraskov", "Leonenko"}):
        hc = funKraskov.hc
    else : 
        print("unknown method... check argument")
        # return a tuple containing 1 element
        h = lambda a, b, c: (numpy.nan, )
    hXkY = hc(x, y, param[0])
    return hXkY[0]
#_______________________________________________________________________________
#_______________________________________________________________________________
def mi(x, y, method="bin", param=(2, )):
    """Compute mutual information of x and y, mi(x ; y)
    
    Syntax 
    
    miXY = mi(x, y, method="bin", param=(2, ))
    
    Input
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    method="bin": {"bin", 'binning'}, {'Kraskov'}, {"Frenzel"}
    param=(2, ): tuple of parameters
    
    Output
    
    miXY: float

    Description
    
    $$ i(x ; y) $$
    wrapper to binning.mi, funKraskov.mi, funFrenzel.mi

    Example

    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.mi(x, y))
    >>> print(dinfo.mi(x, y, method="bin", param=(10, )))
    >>> print(dinfo.mi(x, y, method="Kraskov", param=(10, )))
    >>> print(dinfo.mi(x, y, method="Frenzel", param=(10, "Euclidean")))

        0.202927345006
        0.865491683826
        0.744012468135
        0.744012468135

    See also binning.mi, funKraskov.mi, funFrenzel.mi
    
    """
    if (method in {"bin", "binning"}): 
        miXY = binning.mi(x, y, param[0])[0]
    elif (method in {"Kraskov"}):
        miXY = funKraskov.mi(x, y, k=param[0])
    elif (method in {"Frenzel"}):
        miXY = funFrenzel.mi(x, y, k=param[0], metric=param[1])[0]
    else : 
        print("unknown method... check params")
        miXY = numpy.nan
    return miXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def mic(x, y, z, method="bin", param=(2, )): 
    """Compute conditional mutual information, mic(x ; y | z)

    Syntax
    
    miXYkZ = mic(x, y, z, method="bin", param=(2, )) 
        
    Input 
    
    x: (nDimX, nObs)
    y: (nDimY, nObs)
    z: (nDimZ, nObs)
    method="Frenzel": {"bin", "binning"}, "Frenzel"
    param=(10,): tuple of parameters
    
    Output
    
    miXYkZ: float

    Description

    $$ i(x ; y | z) $$
    wrapper to binning.mic, funFrenzel.mic
    
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.mic(x, y, z))
    >>> print(dinfo.mic(x, y, z, method="Frenzel", param=(10, "Euclidean")))

        0.209132495485
        0.776599296587
        
    See also binning.mic, funKraskov.mic, funFrenzel.mic
    
    """
    if (method in {"bin", "binning"}): 
        miXYkZ = binning.mic(x, y, z, param[0])[0]
    elif (method in {"Frenzel"}):
        miXYkZ = funFrenzel.mic(x, y, z, k=param[0], metric="Euclidean")[0]
    return miXYkZ
#_______________________________________________________________________________
#_______________________________________________________________________________
def te(x, y, p, method="bin", param=(2,)): 
    """ Compute transfert entropy, I(Dx^p -> y^p)

    Syntax
    
    dXY = te(x, y, p, method="bin", param=(2,))

    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    p: order of the model 
    method="bin": {"bin", "binning"}, {"Frenzel"}
    param=(2, ): tuple of parameters
    
    Output
    
    dXY: float
        
    Description

    See Amblard, P. O., & Michel, O. (2014). Causal conditioning and 
    instantaneous coupling in causality graphs. Information Sciences.
    $$ I(Dx \\rightarrow y) \\approx 
        I(x_{t-p}^{t-1}; y_t | y_{t-p}^{t-1}) $$    

    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.te(x, y, 2))
    >>> print(dinfo.te(x, y, 2, method="Frenzel", param=(10, "Euclidean")))    
    
        0.0403500048883
        -0.00142484743689
    
    See also dinfo.mic
    
    """
    x = util.array2D(x)
    y = util.array2D(y)
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    w = yTM1 
    dXY = mic(xTM1[:, p:], y[:, p:], w[:, p:], method, param) 
    return dXY    
#_______________________________________________________________________________
#_______________________________________________________________________________
def tec(x, y, z, p, method="bin", param=(2, )): 
    """ Compute conditional transfert entropy, I(Dx^p -> y^p || Dz^p)
    
    Syntax
    
    dXYkZ = tec(x, y, z, p, method="bin", param=(2, ))
    
    Description
    
    See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
    instantaneous coupling in causality graphs. Information Sciences.
    $$ I(Dx \\rightarrow y \\| Dz) \\approx 
        I(x_{t-p}^{t-1}; y_t | y_{t-p}^{t-1}, z_{t-p}^{t-1}) $$ 
   
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    z: (nDimZ, nObs) or (nObs, )
    p: order of the model 
    method="bin": {"bin", "binning"}, {"Frenzel"}
    param=(2,): tuple of parameters
    
    Output
    
    dXYkZ: float
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.tec(x, y, z, 2))
    >>> print(dinfo.tec(x, y, z, 2, method="Frenzel", param=(10, "Euclidean")))
    
        0.131651216405
        -0.00606132792984

    See also dinfo.mic
    
    """
    x = util.array2D(x)
    y = util.array2D(y)
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    zTM1 = util.getTM1(z, p)
    w = numpy.vstack((yTM1, zTM1)) 
    dXYkZ = mic(xTM1[:, p:], y[:, p:], w[:, p:], method, param)
    return dXYkZ
#_______________________________________________________________________________
#_______________________________________________________________________________
def iie(x, y, p, method="bin", param=(2, )): 
    """Compute instantaneous information exchange, I(x^p -> y^p || Dx^p)
    
    Syntax
    
    dXY = iie(x, y, p, method="bin", param=(2, ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    p: order of the model 
    method="bin": {"bin", "binning"}, {"Frenzel"}
    param={2}: tuple of parameters
    
    Output
    
    dXY: float

    Description
    
    See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
    instantaneous coupling in causality graphs. Information Sciences.
    $$ I(x \\rightarrow y \\| Dx) \\approx 
        I(x_t; y_t | x_{t-p}^{t-1}, y_{t-p}^{t-1}) $$ 

    Example

    imp.reload(dinfo)
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.iie(x, y, 2))
    >>> print(dinfo.iie(x, y, 2, method="Frenzel", param=(10, "Euclidean")))
    
        0.222350062457
        0.262513160772

    See also dinfo.mic
    
    """
    x = util.array2D(x)
    y = util.array2D(y)
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    w = numpy.vstack((xTM1, yTM1)) 
    dXY = mic(x[:, p:], y[:, p:], w[:, p:], method, param)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def iieu(x, y, z, p, method="bin", param=(2, )): 
    """ Compute instantaneous unconditional information exchange, I(x^p -> y^p || Dx^p, Dz^p)
    
    Syntax
    
    dXYkZ = iieu(x, y, z, p, method="bin", param=(2, ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    z: (nDimZ, nObs) or (nObs, )
    p: order of the model 
    method="bin": {"bin", "binning"}, {"Frenzel"}
    param=(2,): tuple of parameters
    
    Output
    
    dXYkZ: float

    Description
    
    See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
    instantaneous coupling in causality graphs. Information Sciences.
    $$ I(x \\rightarrow y \\| Dx, Dz) \\approx 
        I(x_t; y_t | x_{t-p}^{t-1}, y_{t-p}^{t-1}, z_{t-p}^{t-1}) $$ 

    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.iieu(x, y, z, 2))
    >>> print(dinfo.iieu(x, y, z, 2, method="Frenzel", param=(10, "Euclidean")))
    
        0.203840574812
        0.104391830274

    See also dinfo.mic
    
    """
    x = util.array2D(x)
    y = util.array2D(y)
    z = util.array2D(z)
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    zTM1 = util.getTM1(z, p)
    w = numpy.vstack((xTM1, yTM1, zTM1)) 
    dXYkZ = mic(x[:, p:], y[:, p:], w[:, p:], method, param)
    return dXYkZ    
#_______________________________________________________________________________
#_______________________________________________________________________________
def iiec(x, y, z, p, method="bin", param=(2, )): 
    """ Compute instantaneous conditional information exchange, I(x^p -> y^p || Dx^p, z^p)
    
    Syntax
    
    dXYkZ = iiec(x, y, z, p, method="bin", param=(2, ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nDimY, nObs) or (nObs, )
    z: (nDimZ, nObs) or (nObs, )
    p: order of the model 
    method="bin": {"bin", "binning"}, {"Frenzel"}
    param=(2,): tuple of parameters
    
    Output
    
    dXYkZ: float

    Description
    
    See Amblard, P. O., & Michel, O. (2014). Causal conditioning and
    instantaneous coupling in causality graphs. Information Sciences.
    $$ I(x \\rightarrow y \\| Dx, z) \\approx 
        I(x_t; y_t | x_{t-p}^{t-1}, y_{t-p}^{t-1}, z_{t-p}^{t}) $$ 

    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> print(dinfo.iiec(x, y, z, 2))
    >>> print(dinfo.iiec(x, y, z, 2, method="Frenzel", param=(10, "Euclidean")))

    0.135230295235
    0.0677397353161

    See also dinfo.mic
    
    """
    x = util.array2D(x)
    y = util.array2D(y)
    z = util.array2D(z)
    xTM1 = util.getTM1(x, p)
    yTM1 = util.getTM1(y, p)
    zT = util.getT(z, p)
    w = numpy.vstack((xTM1, yTM1, zT)) 
    dXYkZ = mic(x[:, p:], y[:, p:], w[:, p:], method, param)
    return dXYkZ    
#_______________________________________________________________________________
#_______________________________________________________________________________
# Geweke's causal measures
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcm(x, y, z, p, condition="", method="dynamic", 
    kernelMethod="Gaussian", nFold=10, listLambda=[1.], param=([1.], )):
    """Compute Geweke's causal measure, Gx.y, Gx->y, Gx.y|z or Gx->y|z 
    
    Syntax
    
    dXY = gcm(x, y, z, p, condition="", method="dynamic", 
        kernelMethod="Gaussian", nFold=10, listLambda=[1.], param=([1.], ))
    
    Input 
    
    x: (nDimX, nObs)
    y: (nObs, )
    z: (nDimZ, nObs) or []
    p: int, order of the model 
    condition="": {"", "unconditional"}; "conditional"
    method="dynamic": {"dynamic", "d"}; {"instantaneous", 'instant', "d"}
    kernelMethod="Gaussian": "Gaussian"; "linear"
    listLambda=[1.]: list of lambda values
    param=([1.], ): tuple of list of parameters for the kernel
    
    Output
    
    dXY: float

    Description
    
    wrapper to dinfo.gcmd, dinfo.gcmi, dinfo.gcmcd, dinfo.gcmci

    Example 
    
    >>> x = numpy.array([2, 3, 4, 5, 6, 7, 8])
    >>> y = numpy.array([1, 2, 3, 4, 5, 6, 7])
    >>> z = numpy.array([[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, 27]])
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> kernelMethod = "Gaussian"
    >>> r_gcmd = dinfo.gcm(x, y, z, 1, "", "dynamic", kernelMethod, 2, listLambda, param)
    >>> r_gcmi = dinfo.gcm(x, y, z, 1, "", "instant", kernelMethod, 2, listLambda, param)
    >>> r_gcmcd = dinfo.gcm(x, y, z, 1, "conditional", "dynamic", kernelMethod, 2, listLambda, param)
    >>> r_gcmci = dinfo.gcm(x, y, z, 1, "conditional", "instant", kernelMethod, 2, listLambda, param)
    >>> print((r_gcmd, r_gcmi, r_gcmcd, r_gcmci))
    >>> kernelMethod = "linear"
    >>> r_gcmd = dinfo.gcm(x, y, z, 1, "", "dynamic", kernelMethod, 2, listLambda, param)
    >>> r_gcmi = dinfo.gcm(x, y, z, 1, "", "instant", kernelMethod, 2, listLambda, param)
    >>> r_gcmcd = dinfo.gcm(x, y, z, 1, "conditional", "dynamic", kernelMethod, 2, listLambda, param)
    >>> r_gcmci = dinfo.gcm(x, y, z, 1, "conditional", "instant", kernelMethod, 2, listLambda, param)
    >>> print((r_gcmd, r_gcmi, r_gcmcd, r_gcmci))
    
    (-0.31030602699494281, -0.43394609466508832, -0.31868242671114855, -0.1429022653929066)
    (5.3566290601691016, 4.5356931016147923, 1.1006376591277687, 0.5842411064151809)
    
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> kernelMethod = "Gaussian"
    >>> r_gcmd = dinfo.gcm(x, y, z, 2, "", "dynamic", kernelMethod, 2, listLambda, param)
    >>> r_gcmi = dinfo.gcm(x, y, z, 2, "", "instant", kernelMethod, 2, listLambda, param)
    >>> r_gcmcd = dinfo.gcm(x, y, z, 2, "conditional", "dynamic", kernelMethod, 2, listLambda, param)
    >>> r_gcmci = dinfo.gcm(x, y, z, 2, "conditional", "instant", kernelMethod, 2, listLambda, param)
    >>> print((r_gcmd, r_gcmi, r_gcmcd, r_gcmci))

    (0.071283592713089969, 1.4785970563487212, 0.024873850455923541, 3.2455896779735904)
    
    See also dinfo.gcmd, dinfo.gcmi, dinfo.gcmcd, dinfo.gcmci
    
    """
    if (condition == ""): 
        if (method in {"dynamic", "d"}):
            dXY = gcmd(x, y, p, kernelMethod, nFold, listLambda, param)
        elif (method in {"instantaneous", "instant", "i"}): 
            dXY = gcmi(x, y, p, kernelMethod, nFold, listLambda, param)
    elif (condition == "conditional"): 
        if (method in {"dynamic", "d"}):
            dXY = gcmcd(x, y, z, p, kernelMethod, nFold, listLambda, param)
        elif (method in {"instantaneous", "instant", "i"}): 
            dXY = gcmci(x, y, z, p, kernelMethod, nFold, listLambda, param)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmd(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda = [1.], 
    param=([1.], )):
    """Compute Geweke's dynamic causal measure, Gx->y
    
    Syntax
    
    gcmd(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda = [1.], 
        param=([1.], ))
    
    Input 
    
    x: (nDimX, nObs)
    y: (nObs, )
    p: int, order of the model
    kernelMethod="Gaussian": "Gaussian", "linear"
    listLambda=[1.]: list of lambda values
    param=([1.],): tuple of list of parameters for the kernel
    
    Output
    
    dXY: float

    Description
    
    $$ G_{X \\rightarrow Y} = \\frac {\\sigma^2(y_t | y^{t - 1})}
        {\\sigma^2(y_t | y^{t - 1}, x^{t - 1})}$$ 
    wrapper to funGewekeKernel.gcmd
    
    Example 
    
    >>> x = numpy.array([2, 3, 4, 5, 6, 7, 8])
    >>> y = numpy.array([1, 2, 3, 4, 5, 6, 7])
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> kernelMethod = "Gaussian"
    >>> r_gcmdXY = dinfo.gcmd(x, y, 1, kernelMethod, 2, listLambda, param)
    >>> r_gcmdYX = dinfo.gcmd(y, x, 1, kernelMethod, 2, listLambda, param)
    >>> print((r_gcmdXY, r_gcmdYX))
    >>> kernelMethod = "linear"
    >>> r_gcmdXY = dinfo.gcmd(x, y, 1, kernelMethod, 2, listLambda, param)
    >>> r_gcmdYX = dinfo.gcmd(y, x, 1, kernelMethod, 2, listLambda, param)
    >>> print((r_gcmdXY, r_gcmdYX))   
    
        (-0.31030602699494281, -0.44438657625758105)
        (5.3566290601691016, 2.7145343664169874)
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(dinfo.gcmd(x, y, 1, "Gaussian", 2, listLambda, param))
    
        0.0712835927131

    See also funGewekeKernel.gcmd
    """
    dXY = funGewekeKernel.gcmd(x, y, p, kernelMethod, nFold, listLambda, param)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmi(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.], )):
    """Compute Geweke's instantaneous causal measure, Gx.y
    
    Syntax
    
    dXY = gcmi(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
        param=([1.], ))
    
    Input 
    
    x: (nDimX, nObs)
    y: (nObs, )
    p: int, order of the model 
    kernelMethod="Gaussian": "Gaussian", "linear"
    listLambda=[1.]: list of lambda values
    param=([1.], ): tuple of list of parameters for the kernel
    
    Output
    
    dXY: float

    Description
    
    $$ G_{X . Y} = \\frac {\\sigma^2(y_t | y^{t - 1}, x^{t - 1})}
        {\\sigma^2(y_t | y^{t - 1}, x^{t})}$$ 
    wrapper to funGewekeKernel.gcmi

    Example 
    
    >>> x = numpy.array([2, 3, 4, 5, 6, 7, 8])
    >>> y = numpy.array([1, 2, 3, 4, 5, 6, 7])
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> kernelMethod = "Gaussian"
    >>> r_gcmiXY = dinfo.gcmi(x, y, 1, kernelMethod, 2, listLambda, param)
    >>> r_gcmiYX = dinfo.gcmi(y, x, 1, kernelMethod, 2, listLambda, param)
    >>> print((r_gcmiXY, r_gcmiYX))
    >>> kernelMethod = "linear"
    >>> r_gcmiXY = dinfo.gcmi(x, y, 1, kernelMethod, 2, listLambda, param)
    >>> r_gcmiYX = dinfo.gcmi(y, x, 1, kernelMethod, 2, listLambda, param)
    >>> print((r_gcmiXY, r_gcmiYX))

        (-0.31868242671114855, -0.32303729595148267)
        (1.1006376591277687, 1.3300470608009509)
        
    Example

    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(dinfo.gcmi(x, y, 1, "Gaussian", 2, listLambda, param))
   
        0.0248738504559
    
    See also funGewekeKernel.gcmi
    
    """
    dXY = funGewekeKernel.gcmi(x, y, p, kernelMethod, nFold, listLambda, param)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmcd(x, y, z, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.], )):
    """Compute Geweke's dynamic causal measure conditionally to z, Gx->y|z
    
    Syntax
    
    dXYkZ = gcmcd(x, y, z, p, kernelMethod="Gaussian", nFold=10, 
        listLambda=[1.], param=([1.], ))
    
    Input 
    
    x: (nDimX, nObs)
    y: (nObs, )
    z: (nDimZ, nObs)
    p: order of the model 
    kernelMethod="Gaussian": "Gaussian", "linear"
    listLambda=[1.]: list of lambda values
    param=([1.], ): tuple of list of parameters for the kernel
    
    Output
    
    dXYkZ: float

    Description
    
    $$ G_{X \\rightarrow Y || Z} = \\frac{\\sigma^2(y_t | y^{t - 1}, z^{t - 1})}
        {\\sigma^2(y_t | y^{t - 1}, x^{t - 1}, z^{t - 1})}$$ 
    Wrapper to funGewekeKernel.gcmcd 
    
    Example 
    
    >>> x = numpy.array([2, 3, 4, 5, 6, 7, 8])
    >>> y = numpy.array([1, 2, 3, 4, 5, 6, 7])
    >>> z = numpy.array([[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, 27]])
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> kernelMethod = "Gaussian"
    >>> gcmcdXYKZ = dinfo.gcmcd(x, y, z, 1, kernelMethod, 2, listLambda, param)
    >>> gcmcdYXKZ = dinfo.gcmcd(y, x, z, 1, kernelMethod, 2, listLambda, param)
    >>> print(gcmcdXYKZ, gcmcdYXKZ)
    >>> kernelMethod = "linear"
    >>> gcmcdXYKZ = dinfo.gcmcd(x, y, z, 1, kernelMethod, 2, listLambda, param)
    >>> gcmcdYXKZ = dinfo.gcmcd(y, x, z, 1, kernelMethod, 2, listLambda, param)
    >>> print(gcmcdXYKZ, gcmcdYXKZ)

        (-0.31868242671114855, -0.32303729595148267)
        (1.1006376591277687, 1.3300470608009509)
    
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(dinfo.gcmcd(x, y, z, 1, "Gaussian", 2, listLambda, param))
   
        0.0248738504559

    See also funGewekeKernel.gcmcd
    
    """
    dXYkZ = funGewekeKernel.gcmcd(x, y, z, p, kernelMethod, nFold, listLambda, param)
    return dXYkZ
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmci(x, y, z, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.],)):
    """Compute Geweke's instantaneous causal measure conditionally to z, Gx.y|z  
    
    Syntax
    
    dXYkZ = gcmci(x, y, z, p, kernelMethod="Gaussian", nFold=10,    
        listLambda=[1.], param=([1.],))
    
    Input 
    
    x: (nDimX, nObs)
    y: (nObs,)
    z: (nDimZ, nObs)
    p: order of the model 
    kernelMethod="Gaussian": "Gaussian", "linear"
    listLambda=[1.]: list of lambda values
    param=([1.], ): tuple of list of parameters for the kernel
      
    Output
    
    dXYkZ: float
    
    Description
    
    $$ G_{X . Y || Z} = \\frac{\\sigma^2(y_t | y^{t - 1}, x^{t - 1}, z^{t})}
        {\\sigma^2(y_t | y^{t - 1}, x^{t}, z^{t})}$$
    Wrapper to funGewekeKernel.gcmci
    
    Example 
    
    >>> x = numpy.array([2, 3, 4, 5, 6, 7, 8])
    >>> y = numpy.array([1, 2, 3, 4, 5, 6, 7])
    >>> z = numpy.array([[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, 27]])
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> kernelMethod = "Gaussian"
    >>> gcmciXYKZ = dinfo.gcmci(x, y, z, 1, kernelMethod, 2, listLambda, param)
    >>> gcmciYXKZ = dinfo.gcmci(y, x, z, 1, kernelMethod, 2, listLambda, param)
    >>> print((gcmciXYKZ, gcmciYXKZ))
    >>> kernelMethod = "linear"
    >>> gcmciXYKZ = dinfo.gcmci(x, y, z, 1, kernelMethod, 2, listLambda, param)
    >>> gcmciYXKZ = dinfo.gcmci(y, x, z, 1, kernelMethod, 2, listLambda, param)
    >>> print((gcmciXYKZ, gcmciYXKZ))
    
        (-0.1429022653929066, -0.13723766978942556)
        (0.5842411064151809, 0.66713424839386737)
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(dinfo.gcmci(x, y, z, 1, "Gaussian", 2, listLambda, param))
    
        3.24558967797
    
    See alos funGewekeKernel.gcmci

    """
    dXYkZ = funGewekeKernel.gcmci(x, y, z, p, kernelMethod, nFold, listLambda, param)
    return dXYkZ
#_______________________________________________________________________________
#_______________________________________________________________________________
# Gaussian process
#_______________________________________________________________________________
#_______________________________________________________________________________
def gprcm(x, y, p, kernelMethod="Gaussian", listLambda=[1.], param=([1.],)):
    """Compute Gaussian process regression causality measure, gpr(x->y)
    
    Syntax
    
    dXY = gprcm(x, y, p, kernelMethod="Gaussian", listLambda=[1.], param=([1.],))
    
    Input
    
    x: (nDimX, nObs)
    y: (nObs, )
    p: order of the model 
    kernelMethod="Gaussian": "Gaussian; "linear"
    listLambda=[1.]: list of lambda values for optimization
    param=([1.], ): tuple of list of param for optimization
    
    Output
    
    dXY: float

    Description
    
    $$ d_{x \\rightarrow y} = 
        \\mbox{max}_{\\theta_2} \\log{P_2(y(t) | x^{t-1}, y^{t-1})} - 
        \\mbox{max}_{\\theta_1} \\log{P_1(y(t) | y^{t-1})} $$
    wrapper to gpr.gprcm_Gaussian or gpr.gprcm_linear
    
    Example
    
    >>> x = numpy.array([[1., 2., 3., 4., 5., 6., 7.]]) 
    >>> y = numpy.array([[2., 3., 4., 5., 6., 7., 8.]]) 
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> dXY = dinfo.gprcm(x, y, 3, "Gaussian", listLambda, param)
    >>> print(dXY)
    >>> dXY = dinfo.gprcm(x, y, 3, "linear", listLambda)
    >>> print(dXY)
    
        -0.803688982849
        -0.337300953407
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(dinfo.gprcm(x, y, 2, "Gaussian", listLambda, param))
        
        1.5417645692
      
    See also gpr.gprcm_Gaussian, gpr.gprcm_linear
    
    """
    if (kernelMethod == "Gaussian"): 
        dXY = gpr.gprcm_Gaussian(x, y, p, listLambda, listBeta=param[0])
    elif (kernelMethod == "linear"): 
        dXY = gpr.gprcm_linear(x, y, p, listLambda)
    return dXY[0]
#_______________________________________________________________________________
#_______________________________________________________________________________
def gprcmc(x, y, z, p, kernelMethod="Gaussian", listLambda=[1.], param=([1.],)): 
    """Compute Gaussian process regression causality measure conditionally to z, gpr(x->y|z) 
    
    Syntax
    
    dXYkZ = gprcmc(x, y, z, p, kernelMethod="Gaussian", listLambda=[1.], 
        param=([1.],))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nObs, )
    z: (nDimZ, nObs)
    p: order of the model 
    kernelMethod='Gaussian": "Gaussian; "linear"
    listLambda=[1.]: list of lambda values for optimization
    param=([1.], ): tuple of list of param for optimization
    
    Output
    
    dXYkZ: float

    Description
    
    $$ d_{x \\rightarrow y | z} = 
        \\mbox{max}_{\\theta_2} \\log{P_2(y(t) | x^{t-1}, y^{t-1}, z^{t-1})} - 
        \\mbox{max}_{\\theta_1} \\log{P_1(y(t) | y^{t-1}, z^{t-1})} $$
    wrapper to gpr.gprcmc_Gaussian, gpr.gprcmc_linear

    Example
    
    >>> x = numpy.array([[1., 2., 3., 4., 5., 6., 7.]]) 
    >>> y = numpy.array([[2., 3., 4., 5., 6., 7., 8.]]) 
    >>> z = numpy.array([[1., 1., 1., 1., 1., 1., 1.]]) 
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> dXYkZ = dinfo.gprcmc(x, y, z, 3, "Gaussian", listLambda, param)
    >>> print(dXYkZ)
    >>> dXYkZ = dinfo.gprcmc(x, y, z, 3, "linear", listLambda)
    >>> print(dXYkZ)
    
        -0.803688982849
        -0.496189301554
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(dinfo.gprcmc(x, y, z, 2, "Gaussian", listLambda, param))
    
        0.00110640807074
    """
    if (kernelMethod == "Gaussian"): 
        dXYkZ = gpr.gprcmc_Gaussian(x, y, z, p, listLambda, param[0])
    elif (kernelMethod == "linear"): 
        dXYkZ = gpr.gprcmc_linear(x, y, z, p, listLambda)
    return dXYkZ[0]
#_______________________________________________________________________________
