import numpyro
import numpyro.distributions as npdist
import jax.numpy as jnp
from jax import jit
import jax.random
import random
from functools import partial


import pints_jax
import pints_jax.toy as toy
import pints_jax.toy.stochastic

# @jax.jit
def MM(x: jnp.ndarray, initial: jnp.ndarray):
    """
    Michaelis Menten numpyro primitive to generate samples from it.

    :param x: Jax array, serves as time grid.
    :param y: observations.
    :param noise: if True, add noise to the sample. The noise is drawn from the half-normal distribution with
                  variance of 0.1.
    :param sample_k: if True, sample k from a Uniform distribution.

    """
    key = jax.random.PRNGKey(random.randint(0, 100000))
    low = jnp.array([5e-6,0.1,0.1])
    up = jnp.array([5e-5,0.3,0.3])

    # if sample_k:
    k = numpyro.sample("k", npdist.Uniform(low, up),rng_key = key)
    k = k.reshape(3,1)
    # else: k = [1e-5, 0.2, 0.2]

    model = toy.stochastic.MichaelisMentenModel(initial)
    y = model.simulate(k, x)
    # return jnp.asarray(y), jnp.asarray(k)
    return y, k
