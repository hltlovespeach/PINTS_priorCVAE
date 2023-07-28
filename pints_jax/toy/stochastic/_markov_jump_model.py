#
# Markov jump model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import scipy.interpolate as interpolate

import pints

import jax
import jax.numpy as jnp
import jax.scipy
from jax_cosmo import scipy as jcospy
import jax.lax as lax

import random as rnd

from .. import ToyModel


class MarkovJumpModel(pints.ForwardModel, ToyModel):
    r"""
    A general purpose Markov Jump model used for any systems of reactions
    that proceed through jumps.

    A population of N different species is simulated, reacting through M
    different reaction equations.

    Simulations are performed using Gillespie's algorithm [1]_, [2]_:

    1. Sample values :math:`r_0`, :math:`r_1`, from a uniform distribution

    .. math::
        r_0, r_1 \sim U(0,1)

    2. Calculate the time :math:`\tau` until the next single reaction as

    .. math::
        \tau = \frac{-\ln(r)}{a_0}

    where :math:`a_0` is the sum of the propensities at the current time.

    3. Decide which reaction, i, takes place using :math:`r_1 * a_0` and
    iterating through propensities. Since :math:`r_1` is a a value between 0
    and 1 and :math`a_0` is the sum of all propensities, we can find :math:`k`
    for which :math:`s_k / a_0 <= r_2 < s_(k+1) / a_0` where :math:`s_j` is the
    sum of the first :math:`j` propensities at time :math:`t`. We then choose
    :math:`i` as the reaction corresponding to propensity k.

    4. Update the state :math:`x` at time :math:`t + \tau` as:

    .. math::
        x(t + \tau) = x(t) + V[i]

    4. Return to step (1) until no reaction can take place or the process
    has gone past the maximum time.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    x_0
        An N-vector specifying the initial population of each
        of the N species.
    V
        An NxM matrix consisting of stochiometric vectors :math:`v_i`
        specifying the changes to the state, x, from reaction i taking place.
    propensities
        A function from the current state, x, and reaction rates, k, to a
        vector of the rates of each reaction taking place.

    References
    ----------
    .. [1] A Practical Guide to Stochastic Simulations of Reaction Diffusion
           Processes. Erban, Chapman, Maini (2007).
           arXiv:0704.1908v2 [q-bio.SC]
           https://arxiv.org/abs/0704.1908
    .. [2] A general method for numerically simulating the stochastic time
           evolution of coupled chemical reactions. Gillespie (1976).
           Journal of Computational Physics
           https://doi.org/10.1016/0021-9991(76)90041-3
    """
    def __init__(self, x0, V, propensities):
        super(MarkovJumpModel, self).__init__()
        self._x0 = jnp.asarray(x0)
        self._V = V
        self._propensities = propensities

        t = 0
        x = jnp.array(self._x0)
        self.mol_count = [jnp.array(x)]
        self.time = [t]
        # if jnp.any(self._x0 < 0):
        #     raise ValueError('Initial molecule count cannot be negative.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return len(self._V)

    def simulate_raw(self, rates, max_time):
        """ Returns raw times, mol counts when reactions occur. """
        # if len(rates) != self.n_parameters():
        #     raise ValueError(
        #         'This model should have ' + str(self.n_parameters())
        #         + ' parameter(s).')

        # Setting the current propensities and summing them up
        current_propensities = self._propensities(self._x0, rates)
        cp = jnp.asarray(current_propensities)
        prop_sum = jnp.sum(cp)

        # Initial time and count
        t = 0
        # x = jnp.array(self._x0)

        # Run Gillespie SSA, calculating time until next reaction, deciding
        # which reaction, and applying it
        # self.mol_count = [jnp.array(x)]
        # self.time = [t]
        
        arr = [prop_sum, t, max_time, cp]
        def cond(arr):
            return jnp.logical_and(arr[0] > 0, arr[1] <= arr[2])
        
        def body(arr):
            
            cp_1 = jnp.asarray(self._propensities(self._x0, rates))
            x = jnp.array(self._x0)
            V = jnp.asarray(self._V)

            # time = self.time
            # mol_count = self.mol_count

            key_1, key_2 = jax.random.split(jax.random.PRNGKey(rnd.randint(0, 9999)))
            r_1, r_2 = jax.random.uniform(key_1), jax.random.uniform(key_2)
            arr[1] += -jnp.log(r_1) / arr[0]

            def cond_1(sr):
                return sr[0][0] <= r_2 * arr[0]
            
            def body_2(sr):
                idx = sr[1][0]
                ad = cp_1[idx]
                sr = sr.at[1].add(1)
                sr = sr.at[0].add(ad)
                return jnp.asarray(sr)

            srzero = jnp.array([[0],[0]])
            newsr = jax.lax.while_loop(cond_fun=cond_1, body_fun=body_2, init_val=srzero)

            x += V[newsr[1][0] - 1]

            # Calculate new current propensities
            current_propensities = self._propensities(x, rates)
            current_propensities = jnp.asarray(current_propensities)
            arr[0] = jnp.sum(current_propensities)

            # Store new values
            self.time.append(arr[1])
            self.mol_count.append(x)
            jnp.asarray(self.time)
            jnp.asarray(self.mol_count)
            
            return arr

        last = jax.lax.while_loop(cond_fun=cond, body_fun=body, init_val=arr)

        return self.time, self.mol_count

    def sorted_interp(self, x, xp, fp):
        m = x.shape[0]
        # n = xp.shape[0]
        n = len(xp)
        x = jnp.atleast_1d(x)
        j = 0
        xp_0 = xp[0]
        fp_0 = fp[0]

        def inner_fun(args):
            x_i, j = args
            def cond_fun(state):
                is_continuing, *_ = state
                return is_continuing

            def body_fun(state):
                _, _, curr_j, curr_xp_j, curr_fp_j = state

                next_xp_j = xp[curr_j + 1]
                next_fp_j = fp[curr_j + 1]

                cond = x_i > next_xp_j

                def cond_true(_):
                    inner_cond = curr_j + 1 == n - 1

                    def fun_true(_): return False, True, curr_j, next_xp_j, next_fp_j
                    def fun_false(_): return True, False, curr_j + 1, next_xp_j, next_fp_j

                    return lax.cond(inner_cond, fun_true, fun_false, None)

                def cond_false(_):
                    inner_cond = curr_fp_j == next_xp_j
                
                    def fun_true(_):  return False, True, curr_j, next_xp_j, next_fp_j
                    def fun_false(_):  return False, False, curr_j, next_xp_j, next_fp_j

                    return lax.cond(inner_cond, fun_true, fun_false, None)

                return lax.cond(cond, cond_true, cond_false, None)

            _, use_next_fp_j, new_j, *_ = lax.while_loop(cond_fun, body_fun, (True, False, j, xp[j], fp[j]))
            # We don't compute the result inside the loop to allow for seemless backward mode differentiability
            return new_j, lax.cond(use_next_fp_j, 
                                    lambda _: fp[new_j + 1],
                                   lambda _: fp[new_j] + + (fp[new_j + 1] - fp[new_j]) * (x_i - xp[new_j]) / (xp[new_j + 1] - xp[new_j]),
                                   None)

        def body_fun(j, x_i):
            return lax.cond(x_i <= xp_0, lambda *_: (j, fp_0), inner_fun, (x_i, j))

        _, f = lax.scan(body_fun, 0, x)
        return f
        
    def interp(self, x, xp, fp, left=None, right=None, period=None):
        x, xp, fp = map(jnp.asarray, (x, xp, fp)) # this line triggers the leakage error
        if period:
            x = x % period
            xp = xp % period
            i = jnp.argsort(xp)
            xp = xp[i]
            fp = fp[i]
            xp = jnp.concatenate([xp[-1:] - period, xp, xp[:1] + period])
            fp = jnp.concatenate([fp[-1:], fp, fp[1:]])
  
        i = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, len(xp) - 1)
        f = (fp[i - 1] *  (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1])

        if not period:
            if left is None:
                left = fp[0]
            if right is None:
                right = fp[-1]
            f = jnp.where(x < xp[0], left, f)
            f = jnp.where(x > xp[-1], right, f)
        return f

    def interpolate_mol_counts(self, time_interp, mol_count, output_times):
        """
        Takes raw times and inputs and mol counts and outputs interpolated
        values at output_times
        """
        if len(time_interp) == 0:
            raise ValueError('At least one time must be given.')
        if len(time_interp) != len(mol_count):
            raise ValueError('The number of entries in time must match mol_count')

        # Check output times
        output_times = jnp.asarray(output_times)
        # if jnp.logical_not(jnp.all(output_times[1:] >= output_times[:-1])) :
        #     raise ValueError('The output_times must be non-decreasing.')

        # Interpolate as step function, decreasing mol_count by 1 at each
        # reaction time point.
        if len(time_interp) == 1:
            # Need at least 2 values to interpolate
            return jnp.ones(len(output_times)) * mol_count[0]
        else:
            # Note: Can't use fill_value='extrapolate' here as:
            #  1. This require scipy >= 0.17
            #  2. There seems to be a bug in some scipy versions

            values = self.sorted_interp(output_times, time_interp, mol_count)

            # values = self.interp(output_times, time_interp, mol_count)
            # triggers leakage error

            # values = jcospy.interpolate.interp(output_times, time_interp, mol_count)
            # triggers Error: unsupported operand type(s) for -: 'BatchTracer' and 'list'
            # Guess: interprets time_interp as a list

            # values = jax.numpy.interp(output_times, time_interp, mol_count)
            # interp_function = jax.scipy.interpolate.RegularGridInterpolator(time_interp, mol_count,fill_value=jnp.nan, bounds_error=False)
            # values = interp_func(output_times)
        
        # At any point past the final time, repeat the last value
        values[output_times >= time_interp[-1]] = mol_count[-1]
        values[output_times < time_interp[0]] = mol_count[0]

        return values

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        times = jnp.asarray(times)
        # if jnp.any(times < 0):
        #     raise ValueError('Negative times are not allowed.')

        # Run Gillespie algorithm
        time, mol_count = self.simulate_raw(parameters, jnp.max(times))

        print(time)
        # print(mol_count)

        # Interpolate and return
        return self.interpolate_mol_counts(time, mol_count, times)





# original while loop in simulate_raw
 # while (prop_sum > 0) and (t <= max_time):
        #     r_1, r_2 = jnp.random.uniform(0, 1), jnp.random.uniform(0, 1)
        #     t += -jnp.log(r_1) / prop_sum
        #     s = 0
        #     r = 0
        #     while s <= r_2 * prop_sum:
        #         s += current_propensities[r]
        #         r += 1
        #     x += self._V[r - 1]

        #     # Calculate new current propensities
        #     current_propensities = self._propensities(x, rates)
        #     current_propensities = jnp.asarray(current_propensities)
        #     prop_sum = jnp.sum(current_propensities)

        #     # Store new values
        #     time.append(t)
        #     mol_count.append(jnp.copy(x))

# cond_1 = (sr[0] <= r_2 * prop_sum)
# def true_fun_1(sr):
#     sr[0] += current_propensities[sr[1]]
#     sr[1] += 1

# def false_fun_1(sr):
#     return sr

# f_1 = jax.lax.cond(pred=cond_1, true_fun=true_fun_1, false_fun=false_fun_1, operands=sr)

# jax.lax.scan(f_1,)
# def add_index(s,r):

# jax.lax.scan(f, init, xs, length)
