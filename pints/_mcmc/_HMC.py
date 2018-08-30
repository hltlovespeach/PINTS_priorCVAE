#
# Hamiltonian Monte Carlo method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class HMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`

    Implements Hamiltonian Monte Carlo as described in [1].

    Uses a physical analogy of a particle moving across a landscape under
    Hamiltonian dynamics to aid efficient exploration of parameter space.
    Introduces an auxilary variable -- the momentum (``p_i``) of a particle
    moving in dimension ``i`` of negative log posterior space -- which
    supplements the position (``q_i``) of the particle in parameter space. The
    particle's motion is dictated by solutions to Hamilton's equations,

        dq_i/dt =   partial_d H/partial_d p_i,
        dp_i/dt = - partial_d H/partial_d q_i.

    The Hamiltonian is given by,

        H(q,p) =       U(q)       +        KE(p)
               = -log(p(q|X)p(q)) + Sigma_i=1^d p_i^2/2m_i,

    where ``d`` is the dimensionality of model and ``m_i`` is the 'mass' given
    to each particle (often chosen to be 1 as default).

    To numerically integrate Hamilton's equations, it is essential to use a
    sympletic discretisation routine, of which the most typical approach is
    the leapfrog method,

        p_i(t + epsilon) = p_i(t) + epsilon dp_i(t)/dt
                         = p_i(t) - epsilon partial_d U(q)/d_q_i,
        q_i(t + epsilon) = q_i(t) + epsilon dq_i(t)/dt
                         = q_i(t) + epsilon p_i(t) / m_i.

    [1] MCMC using Hamiltonian dynamics
    Radford M. Neal, Chapter 5 of the Handbook of Markov Chain Monte
    Carlo by Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng.

    """
    def __init__(self, x0, log_pdf, sigma0=None):
        super(HMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Current point and proposed point
        self._current = None
        self._current_log_pdf = None
        self._proposed = None

        # Iterations, acceptance monitoring, and leapfrog iterations
        self._iterations = 0
        self._acceptance = 0
        self._leapfrog_iteration = -1

        # Default scale for momentum proposals
        self._momentum_sigma = np.identity(self._dimension)

        # Default integration step size for leapfrog algorithm
        self._epsilon = 0.2

        # Default number of leapfrog steps
        self._leapfrog_steps = 20

        # Default masses
        self._mass = np.ones(self._dimension)

        # Default threshold for Hamiltonian divergences
        # (currently set to match Stan)
        self._hamiltonian_threshold = 10**3

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new point
        if self._proposed is None:  #TODO Change to _asked_point or something

            if self._leapfrog_iteration < 0:
                # Sample initial momentum
                q = self._current
                p = np.random.multivariate_normal(
                    np.zeros(self._dimension), self._momentum_sigma)
                self._current_p = p

                # Leapfrog algorithm
                #TODO Move to tell
                p = p - self._epsilon * grad_U(q) / 2

                #TODO Update leapfrog iteration in tell()

            elif self._leapfrog_iteration < self._leapfrog_steps:

                # Make a full step for the position
                q = q + self._epsilon * p

                # Make a full step for the momentum, except at end of
                # trajectory
                if(i != self._leapfrog_steps):
                    #TODO Move to tell
                    p = p - self._epsilon * grad_U(q)

            else:
                #TODO Update this
                self._proposed = q
                # Make a half step for momentum at the end.
                p = p - self._epsilon * grad_U(q) / 2

                # Negate momentum at end of trajectory to make the proposal
                # symmetric
                self._proposed_p = -p

                # Set as read-only
                self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def epsilon(self):
        """
        Returns the step size for the leapfrog algorithm.
        """
        return self._epsilon

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Propose x0 as first point
        self._current = None
        self._current_log_pdf = None
        self._proposed = self._x0

        # Acceptance rate monitoring
        self._iterations = 0
        self._acceptance = 0

        # Set initial momentum proposals to None
        self._current_p = None
        self._proposed_p = None

        # Create a vector of divergent iterations
        self._divergent = np.asarray([], dtype='int')

        # Update sampler state
        self._running = True

    def leapfrog_steps(self):
        """
        Returns the number of leapfrog steps to carry out for each iteration.
        """
        return self._leapfrog_steps

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')
        #TODO Add leapfrog iteration

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)
        #TODO Add leapfrog iteration

    def momentum_sigma(self):
        """
        Sets the standard deviation for the multivariate normal distribution
        used to propose new momentum values.
        """
        return np.array(self._momentum_sigma, copy=True)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Hamiltonian Monte Carlo'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def set_epsilon(self, epsilon):
        """
        Sets the step size for the leapfrog algorithm.
        """
        if epsilon <= 0:
            raise ValueError(
                'Step size for leapfrog algorithm must be greater than zero.')
        self._epsilon = epsilon

    def set_leapfrog_steps(self, steps):
        """
        Sets the number of leapfrog steps to carry out for each iteration.
        """
        if not isinstance(steps, int):
            raise TypeError('Number of steps must be an integer.')
        if steps < 1:
            raise ValueError('Number of steps must exceed 0.')
        self._leapfrog_steps = steps

    def set_momentum_sigma(self, momentum_sigma):
        """
        Sets the standard deviation for the multivariate normal distribution
        used to propose new momentum values.
        """
        if momentum_sigma.shape != (self._dimension, self._dimension):
            raise ValueError(
                'Multivariate normal for momentum proposals must have shape ('
                + str(self._dimension) + ', ' + str(self._dimension) + ').')

        # Check if positive definite
        if not np.all(np.linalg.eigvals(momentum_sigma) > 0):
            raise ValueError(
                'Multivariate normal for momentum proposals must be positive'
                ' definite.')

        self._momentum_sigma = momentum_sigma

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Clear proposal
            self._proposed = None

            # Return first point for chain
            return self._current

        current_K = np.sum(self._current_p**2 / (2 * self._mass))
        proposed_K = np.sum(self._proposed_p**2 / (2 * self._mass))

        # Check for divergent iterations by testing whether the Hamiltonian
        # difference is above a threshold
        test = fx + proposed_K - (self._current_log_pdf + current_K)
        if test > self._hamiltonian_threshold:
            self._divergent = np.append(self._divergent, self._iterations)

        # Accept or reject the state at end of trajectory, returning either the
        # position at the end of the trajectory or the initial position
        r = np.exp(self._current_log_pdf - fx + current_K - proposed_K)
        if np.random.rand() < r:
            accepted = 1
            self._current = self._proposed
            self._current_log_pdf = fx
        else:
            accepted = 0

        # Update acceptance rate (only used for output!)
        self._iterations += 1
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Return current position
        return self._current

