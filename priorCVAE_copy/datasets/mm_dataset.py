import random as rnd

import jax
import jax.numpy as jnp
from jax import random
from jax import jit

import pints_jax
import pints


class MMDataset:
    """
    Simulate data wrt the Michaelis Menten model. Note: Currently the data is only generated with dimension as 1.
    """

    def __init__(self, x_lim_low: int = 0, x_lim_high: int = 24, n_data: int=100):
        """
        :param x_lim_low: lower limit of the time-interval.
        :param x_lim_high: upper limit if the time-interval.
        """
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.n_data = n_data
        self.times = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)
        self.model = pints_jax.toy.stochastic.MichaelisMentenModel(jnp.array([1e4, 2e3, 2e4, 0]))

    def one_sample(self,input,idx):
        params = jnp.asarray(input[idx,:])
        # print(params)
        mm = self.model.simulate(params, self.times)
        # print(lv)
        return input, mm
        
    def simulatedata(self, n_samples: int = 1000) -> [jnp.ndarray, jnp.ndarray,jnp.ndarray]:
        """
        Simulate data from the LV model.

        :param n_samples: number of samples.

        :returns:
            - time grid, x
            - initial population density+parameter values, 2+4 dimensional
            - population density simulations, (n_data,2)
        """
        rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        params = jax.random.uniform(rng_key, shape=(n_samples,3))
        params = params.at[:,0].multiply(1/50)
        params = params.at[:,1].multiply(1/2)
        params = params.at[:,2].multiply(1/2)
        xs = jnp.asarray(range(n_samples))
        mm_samples = jax.lax.scan(self.one_sample, params, xs)

        return self.times.repeat(n_samples).reshape(self.times.shape[0],n_samples).transpose(), params, jnp.asarray(mm_samples[1:]).reshape(n_samples,self.n_data,4)