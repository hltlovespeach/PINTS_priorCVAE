import random as rnd

import jax
import jax.numpy as jnp
from jax import random
from jax import jit

import pints_jax
import pints


class LVDataset:
    """
    Simulate data wrt the Lotka Volterra model. Note: Currently the data is only generated with dimension as 1.
    """

    def __init__(self, x_lim_low: int = 0, x_lim_high: int = 20, n_data: int=30):
        """
        :param x_lim_low: lower limit of the time-interval.
        :param x_lim_high: upper limit if the time-interval.
        """
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.n_data = n_data
        self.times = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)
        self.model = pints_jax.toy.LotkaVolterraModel()

    def one_sample(self,input,idx):
        params = jnp.asarray(input[idx,:])
        # print(params)
        lv = pints_jax.toy.LotkaVolterraModel().simulate(params, self.times)
        print(lv)
        return input, lv
        
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
        
        
        # params = jax.random.uniform(rng_key, shape=(1,4))
        # params.at[1].multiply(2)
        # params.at[3].multiply(2)
        # params = params[0]
        # params = params*jnp.array([1,2,1,2])
        # print(params)

        # lv_sample = pints_jax.toy.LotkaVolterraModel().simulate(params, self.times)

        params = jax.random.uniform(rng_key, shape=(n_samples,4))
        params = params.at[:,1].multiply(2)
        params = params.at[:,3].multiply(2)
        xs = jnp.asarray(range(n_samples))
        lv_samples = jax.lax.scan(self.one_sample, params, xs)


        # times = jnp.tile(x,(n_samples,1))
        # params = jnp.zeros((n_samples,4))
        # lv_samples = jnp.zeros((n_samples,30,2))
        # for i in range(1000):
        #     _,b,c = generator.simulatedata(1)
        #     params[i,:] = b
        #     lv_samples[i,:,:] = c

        # lv_samples = jnp.apply_along_axis(self.fix_times, 1, params)

        return self.times.repeat(n_samples).reshape(self.times.shape[0],n_samples).transpose(), params, jnp.asarray(lv_samples[1:]).reshape(n_samples,self.n_data,2)