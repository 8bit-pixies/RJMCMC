"""
Bayesian optimization

Chapman's code for a simple bayesian optimization
that works...(hopefully)

This code should be self contained and runs simple example

assumes we are minimizing...
"""

import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize

kernel = gp.kernels.Matern()
model = gp.GaussianProcessRegressor(kernel) # can impose covariance structure?

n_rand = 25
n_iters = 100 # want a way to be by iterations or time - while loop?

# obvious solution is x = 0
# place bound on -10, 10
x_dim = 2

loss_function = lambda x: np.sum(np.atleast_2d(x)**2, 1)

x_start = np.random.uniform(-10.0, 10.0, (5, x_dim))
y_start = loss_function(x_start) # should always assume the function is vectorise, like model.predict

x_all = x_start.copy()
y_all = y_start.copy()

"""
single run...

if x_start is less than 2d -> x_start[:,np.newaxis]
"""

for idx in range(n_iters):
    if idx % 5 == 0:
        print(idx)
        print(y_all[-10:])
    model = model.fit(x_all, y_all)
    
    # calculate expected improvement
    # step 1 generate many many random points    
    x_random = np.random.uniform(-10.0, 10.0, (n_rand, x_dim))
    mu, sigma = model.predict(x_random, return_std=True)
    
    # enter some formulas
    loss_optimum = np.min(y_start)
    
    # with np.errstate(divide='ignore')
    gamma_x = (loss_optimum - mu)/sigma
    expected_improvement = sigma*(gamma_x * (norm.cdf(gamma_x)) + norm.pdf(gamma_x))
    
    # now actually score the loss
    # as this might be expensive be conservative.
    x_next = x_random[np.argmin(expected_improvement)]
    y_random = loss_function(np.atleast_2d(x_next))
    
    # append this one!...
    y_next = loss_function(x_next)
    
    # ...output y_next, x_next
    x_all = np.vstack([x_all, x_next])
    y_all = np.concatenate([y_all, y_next])

# get best...
x_all[np.argmin(y_all)]


def bayesian_optimization(n_iters, loss_function):
    """
    use gaussian process to optimise loss function "loss_function"
    """
    
    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel)


    