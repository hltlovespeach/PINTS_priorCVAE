"""
Gaussian process dataset.

"""

from typing import Union

import random as rnd

import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from numpyro.infer import Predictive

from priorCVAE_copy.priors import GP, Kernel, SquaredExponential, MM
import torch.utils.data as data

class GPDataset:
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, kernel: Kernel, n_data: int = 400, x_lim_low: int = 0,
                 x_lim_high: int = 1, sample_lengthscale: bool = False):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param n_data: number of data points in the interval.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        """
        self.n_data = n_data
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)

    def simulatedata(self, n_samples: int = 10000) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the GP.

        :param n_samples: number of samples.

        :returns:
            - interval of the function evaluations, x, with the shape (num_samples, x_limit).
            - GP draws, f(x), with the shape (num_samples, x_limit).
            - lengthscale values.
        """
        rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        gp_predictive = Predictive(GP, num_samples=n_samples)
        all_draws = gp_predictive(rng_key, x=self.x, kernel=self.kernel, jitter=1e-5,
                                  sample_lengthscale=self.sample_lengthscale)

        ls_draws = jnp.array(all_draws['ls'])
        gp_draws = jnp.array(all_draws['y'])

        return self.x.repeat(n_samples).reshape(self.x.shape[0], n_samples).transpose(), gp_draws, ls_draws

import pints_jax 
# import pints.toy as toy
# import pints.toy.stochastic
import numpyro
import numpyro.distributions as npdist
import time

class MMDataset:
    """
    Generate Michaelis Menten draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_data points.
    This model has 3 parameter values, with 4 different molecules and an intial molecule count.
        - X1+X2 -> X3 with rate k1
        - X3 -> X1+X2 with rate k2
        - X3 -> X2+X4 with rate k3

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, n_data: int = 100, x_lim_low: int = 0,
                 x_lim_high: int = 24):
        """
        Initialize the dataset class.

        :param n_data: number of data points in the interval.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param sample_k: whether to sample k (reaction rates) or not. Defaults to False.
        """
        self.n_data = n_data
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        # self.sample_k = sample_k
        self.x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)

    def simulatedata(self, n_samples: int = 10000, initial: list = [1e4, 2e3, 2e4, 0]) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the MM process.

        :param n_samples: number of samples.

        :returns:
            - x, time grid, with the shape (num_samples, x_limit,4).
            - mm draws, f(x), with the shape (num_samples, x_limit,4).
            - k, parameter values, with the shape (num_samples, 3).
        """
        # rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        # self.sample_k = sample_k
    
        # model = toy.stochastic.MichaelisMentenModel(initial)
        # times = self.x
        # start_time = time.time()

        k_draws = jnp.zeros((n_samples,3,1))
        mm_draws = jnp.zeros((n_samples, self.n_data, 4))
        # mid_time = time.time()
        for i in range(n_samples):
            y,k = MM(x=self.x,initial=jnp.array([1e4, 2e3, 2e4, 0]))
        #     values = model.simulate(k, times)   # shape is (n_data,4)
            k_draws = k_draws.at[i,:,:].set(k.reshape(3,1))
            mm_draws = mm_draws.at[i,:,:].set(y)
        # end_time = time.time()

        # rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))
        
        # mm_predictive = Predictive(MM, num_samples=n_samples)
        # all_draws = mm_predictive(rng_key, x=self.x, initial=initial, sample_k=self.sample_k)

        # k_draws = jnp.array(all_draws['k'])
        # mm_draws = jnp.array(all_draws['y'])

        # print(f"time one is {mid_time - start_time} and time two is {end_time-mid_time}")

        x = self.x.repeat(n_samples).reshape(self.x.shape[0], n_samples).transpose()
        x = x.repeat(4).reshape(n_samples,self.n_data,4)
        return x, mm_draws, k_draws