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
        lv = pints_jax.toy.LotkaVolterraModel().simulate(params, self.times)
        # key, params = input
        # unfinished = True
        # while unfinished:
        #     # rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))
        #     param = jax.random.uniform(key, shape=(1,4))
        #     lv = pints_jax.toy.LotkaVolterraModel().simulate(param[0,:], self.times)
        #     # if jnp.sum(jnp.isnan(lv)) == 0:
        #     #     unfinished = False
        #     #     params.at[idx,:].set(param)
        #     unfinished = not jnp.sum(jnp.isnan(lv)) == 0
        new_lv = jnp.nan_to_num(lv,copy=True, nan=0, posinf=50, neginf=0)
        new_lv = jnp.where(new_lv<50, new_lv, 50)
        new_lv = jnp.where(new_lv>0, new_lv, 0)

        zero = jnp.array([0])
        x,y = jnp.concatenate((new_lv[:,0],zero)), jnp.concatenate((new_lv[:,1],zero))
        rng = jnp.array(range(self.n_data+1))

        id = jnp.min(jnp.where(x==50,rng,30))
        x = jnp.where(rng>=id, 50, x)

        ix = jnp.min(jnp.where(y==50,rng,30))
        y = jnp.where(rng>=ix, 50, y)

        # new_lv = jnp.zeros((30,2))
        new_lv = new_lv.at[:,0].set(x[:self.n_data])
        new_lv = new_lv.at[:,1].set(y[:self.n_data])

        # new_lv = jnp.concatenate((x[:self.n_data],y[:self.n_data]),1)
        # print(new_lv.shape)
        return input ,new_lv
        
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

        # params = jax.random.uniform(rng_key, shape=(n_samples,4))
        # consult https://eurekastatistics.com/beta-distribution-pdf-grapher/
        params = jax.random.beta(rng_key,1.25,2,shape=(n_samples,4))
        params = params.at[:,1:].set([0.15,1,.3])
        # params = params.at[:,1].multiply(2)
        # params = params.at[:,3].multiply(2)

        # params = jnp.zeros((n_samples,4))
        xs = jnp.asarray(range(n_samples))
        # input = (rng_key,params)
        lv_samples = jax.lax.scan(self.one_sample, params, xs)
        
        # params = params[:,jnp.newaxis,jnp.newaxis,:]
        # params = jnp.tile(params, (1, self.n_data, 2, 1))
      

        # times = jnp.tile(x,(n_samples,1))
        # params = jnp.zeros((n_samples,4))
        # lv_samples = jnp.zeros((n_samples,30,2))
        # for i in range(1000):
        #     _,b,c = generator.simulatedata(1)
        #     params[i,:] = b
        #     lv_samples[i,:,:] = c

        # lv_samples = jnp.apply_along_axis(self.fix_times, 1, params)

        return self.times.repeat(n_samples).reshape(
            self.times.shape[0],n_samples).transpose(), jnp.asarray(
                lv_samples[1:]).reshape(n_samples,self.n_data,2,1), params