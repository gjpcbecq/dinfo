#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Contains Geweke's measures using kernel functions

Contain

gcmcd - Geweke's causality measure conditional dynamic
gcmci - Geweke's causality measure conditional instantaneous
gcmd - Geweke's causality measure dynamic
gcmi - Geweke's causality measure instantaneous

Reference

Amblard, P.-O.; Vincent, R.; Michel, O. J. and Richard, C.,``Kernelizing Geweke's measures of granger causality'', Machine Learning for Signal Processing (MLSP), 2012 IEEE International Workshop on, 2012, 1-6

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
import kernel
import numpy
#_______________________________________________________________________________
def gcmd(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.], )): 
    """Compute Geweke's dynamic causal measure on y from x with kernel using n
    fold cross-validation
    
    Syntax
    
    dXY = gcmd(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
        param=([1.], ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nObs, ) 1 dimension
    p: int, order of the model
    kernelMode: "Gaussian", "linear"
    nFold=2: int, number of parts in cross-validation
    listLambda: (1, nLambda), list of float, for lambda parameter
    param: tuple of list of parameters
    
    Output
    
    dXY: float
    
    Description
    
    $$ G_{X \\rightarrow Y} = \\frac {\\sigma^2(y_t | y^{t - 1})}
        {\\sigma^2(y_t | y^{t - 1}, x^{t - 1})}$$ 
    
    Example
    
    >>> x = numpy.arange(1, 101)
    >>> y = numpy.arange(0, 100)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([1, 10, 50], )
    >>> print(funGewekeKernel.gcmd(x, y, 1, "Gaussian", 2, listLambda, param))
    >>> print(funGewekeKernel.gcmd(y, x, 1, "Gaussian", 2, listLambda, param))
    >>> print(funGewekeKernel.gcmd(x, y, 1, "linear", 2, listLambda, param))
    >>> print(funGewekeKernel.gcmd(y, x, 1, "linear", 2, listLambda, param))

        -0.718332623526
        -0.722589415946
        11.818889772
        9.58516115382

    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(funGewekeKernel.gcmd(x, y, 1, "Gaussian", 2, listLambda, param))
    
        0.0712835927131

    """
    xtm1 = util.getTM1(x, p)
    ytm1 = util.getTM1(y, p)
    # num
    w = ytm1
    num = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # den
    w = numpy.vstack((ytm1, xtm1))
    den = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # final
    dXY = num / den
    dXY = numpy.log(dXY)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmi(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.], )): 
    """Compute Geweke's instantaneous causal measure on y from x with kernel 
    using n fold cross-validation
    
    Syntax
    
    dXY = gcmi(x, y, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
        param=([1.], ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nObs, )
    p: int, order of the model
    listLambda: (1, nLambda), list of float, for lambda parameter
    param: tuple of list of parameters
    
    Output
    
    dXY: float
    
    Description
    
    $$ G_{x . y} = \\frac{\\sigma^2(y_t | y^{t - 1}, x^{t - 1})} 
        {\\sigma^2(y_t | y^{t - 1}, x^{t})} $$ 

    Example
    
    >>> x = numpy.arange(1, 101)
    >>> y = numpy.arange(0, 100)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([1, 10, 50], )
    >>> print(funGewekeKernel.gcmi(x, y, 1, "Gaussian", 2, listLambda, param))
    >>> print(funGewekeKernel.gcmi(y, x, 1, "Gaussian", 2, listLambda, param))

        -0.324724189034
        -0.325758585325
        
    Example
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(funGewekeKernel.gcmi(x, y, 1, "Gaussian", 2, listLambda, param))

        1.47859705635
        
    """
    xtm1 = util.getTM1(x, p)
    ytm1 = util.getTM1(y, p)
    xt = util.getT(x, p)
    # num
    w = numpy.vstack((ytm1, xtm1))
    num = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # den
    w = numpy.vstack((ytm1, xt))
    den = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # final
    dXY = num / den
    dXY = numpy.log(dXY)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmcd(x, y, z, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.], )): 
    """Compute Geweke's conditional dynamic causal measure on y from x with 
    kernel using n fold cross-validation
    
    Syntax
    
    dXY = gcmcd(x, y, z, p, kernelMethod="Gaussian", nFold=10, 
        listLambda=[1.], param=([1.], ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nObs, )
    p: int, order of the model
    listLambda: (1, nLambda), list of float, for lambda parameter
    param: tuple of list of parameters
    
    Output
    
    dXY: float
    
    Description
    
    $$ G_{x \\rightarrow y || z} = \\frac{\\sigma^2(y_t | y^{t - 1}, z^{t - 1})} 
     {\\sigma^2(y_t | y^{t - 1}, x^{t - 1}, z^{t - 1})} $$ 
    
    Example
    
    >>> x = numpy.arange(1, 101)
    >>> y = numpy.arange(0, 100)
    >>> z = numpy.ones(100)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([1, 10, 50], )
    >>> print(funGewekeKernel.gcmcd(x, y, z, 1, "Gaussian", 2, listLambda, param))
    >>> print(funGewekeKernel.gcmcd(y, x, z, 1, "Gaussian", 2, listLambda, param))

        -0.718332623526
        -0.722589415946
       
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(funGewekeKernel.gcmcd(x, y, z, 1, "Gaussian", 2, listLambda, param))
        
        0.0248738504559
   
    """
    xtm1 = util.getTM1(x, p)
    ytm1 = util.getTM1(y, p)
    ztm1 = util.getTM1(z, p)
    # num
    w = numpy.vstack((ytm1, ztm1))
    num = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # den
    w = numpy.vstack((ytm1, xtm1, ztm1))
    den = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # final
    dXY = num / den
    dXY = numpy.log(dXY)
    return dXY
#_______________________________________________________________________________
#_______________________________________________________________________________
def gcmci(x, y, z, p, kernelMethod="Gaussian", nFold=10, listLambda=[1.], 
    param=([1.], )): 
    """Compute Geweke's conditional instantaneous causal measure on y from x
    with kernel using n fold cross-validation
    
    Syntax
    
    dXY = gcmci(x, y, z, p, kernelMethod="Gaussian", nFold=10, 
        listLambda=[1.], param=([1.], ))
    
    Input
    
    x: (nDimX, nObs) or (nObs, )
    y: (nObs, )
    p: int, order of the model
    listLambda: (1, nLambda), list of float, for lambda parameter
    param: tuple of list of parameters
    
    Output
    
    dXY: float
        
    Description
    
    $$ G_{x . y || z} = \\frac{\\sigma^2(y_t | y^{t - 1}, x^{t - 1}, z^{t})} 
     {\\sigma^2(y_t | y^{t - 1}, x^{t}, z^{t})} $$ 
    
    Example
    
    >>> x = numpy.arange(1, 101)
    >>> y = numpy.arange(0, 100)
    >>> z = numpy.ones(100)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([1, 10, 50], )
    >>> print(funGewekeKernel.gcmci(x, y, z, 1, "Gaussian", 2, listLambda, param))
    >>> print(funGewekeKernel.gcmci(y, x, z, 1, "Gaussian", 2, listLambda, param))
    
        -0.324724189034
        -0.325758585325    
    
    >>> numpy.random.seed(1)
    >>> (x, y, z) = model.GaussianXYZ(100, 0.9, 0.5, 0.1)
    >>> listLambda = [0.01, 0.1, 1]
    >>> param = ([0.1, 1., 10.], )
    >>> print(funGewekeKernel.gcmci(x, y, z, 1, "Gaussian", 2, listLambda, param))

        3.24558967797
    
    """
    xtm1 = util.getTM1(x, p)
    ytm1 = util.getTM1(y, p)
    ztm1 = util.getTM1(z, p)
    xt = util.getT(x, p)
    zt = util.getT(z, p)
    
    # num
    w = numpy.vstack((ytm1, xtm1, zt))
    num = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # den
    w = numpy.vstack((ytm1, xt, zt))
    den = kernel.mspe_crossfold_search(y[p:], w[:, p:], kernelMethod, nFold, 
        listLambda, param)[0]
    # final
    dXY = num / den
    dXY = numpy.log(dXY)
    return dXY
#_______________________________________________________________________________

