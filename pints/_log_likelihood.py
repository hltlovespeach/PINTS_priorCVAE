#
# Log-likelihood functions
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import pints
import numpy as np

class LogLikelihood(object):
    """
    Represents a log-likelihood that can be used by stochastic methods.
    
    Arguments:
    
    ``problem``
        The time-series problem this log-likelihood is defined for.

    """
    def __init__(self, problem, prior=None):
        self._problem = problem
        # Cache some problem variables
        self._values = problem.values()
        self._times = problem.times()
        self._dimension = problem.dimension()

    def __call__(self, x):
        raise NotImplementedError
        
    def dimension(self):
        """
        Returns the dimension of the space this likelihood is defined on.
        """
        return self._dimension

class BayesianLogLikelihood(LogLikelihood):
    """
    Calculates a log-likelihood based on a (conditional) :class:`LogLikelihood`
    and a class:`Prior`.
    
    The returned value will be `log(prior(x)) + log_likelihood(x|problem)`.
    If `prior(x) == 0` the method always returns `-inf`, regardless of the
    value of the log-likelihood (which will not be evaluated).
    
    """
    def __init__(self, prior, log_likelihood):
        # Check arguments
        if not isinstance(prior, pints.Prior):
            raise ValueError('Prior must extend pints.Prior')
        if not isinstance(log_likelihood, pints.LogLikelihood):
            raise ValueError('Log-likelihood must extends pints.LogLikelihood')
        self._prior = prior
        self._log_likelihood = log_likelihood
        
        # Check dimension
        self._dimension = self._prior.dimension()
        if self._log_likelihood.dimension() != self._dimension:
            raise ValueError('Given prior and log-likelihood must have same'
                ' dimension')

    def __call__(self, x):
        # Evaluate prior first, assuming this is very cheap
        prior = self._prior(x)
        if prior == 0:
            return float('-inf')
        # Take log and add conditional log-likelihood
        return np.log(prior) + self._log_likelihood(x)

class KnownNoiseLogLikelihood(LogLikelihood):
    """
    Calculates a log-likelihood assuming independent normally-distributed noise
    at each time point, using a known value for the standard deviation (sigma)
    of that noise.
    """
    def __init__(self, problem, sigma):
        super(KnownNoiseLogLikelihood, self).__init__(problem)
        # Check sigma
        self._sigma = float(sigma)
        if self._sigma <= 0:
            raise ValueError('Standard deviation must be greater than zero.')
        # Calculate parts
        self._offset = -len(self._times) * np.log(self._sigma)
        self._multip = -1 / float(2 * self._sigma**2)
        
    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        return self._offset + self._multip * np.sum(error**2)

class UnknownNoiseLogLikelihood(LogLikelihood):
    """
    Calculates a log-likelihood assuming independent normally-distributed noise
    at each time point, and adds a parameter representing the standard
    deviation (sigma) of that noise.
    """
    def __init__(self, problem):
        super(UnknownNoiseLogLikelihood, self).__init__(problem)
        # Add sneaky parameter to end of list!
        self._dimension = problem.dimension() + 1
        self._size = len(self._times)
        
    def __call__(self, x):
        error = self._values - self._problem.evaluate(x[:-1])
        return -self._size * np.log(x[-1]) - np.sum(error**2) / (2 * x[-1]**2)
    