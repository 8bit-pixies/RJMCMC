"""
mcmc.py

test script for MH algorithm for some rather silly example

states -> some number on the real number line. 

q(x | x') = q(x' | x) for _detailed balance condition_?
evaluation: p(x) = some objective function (say tree splits)

in this code we will consider the optimal choice of tree splits 
for unlimited splits. Obviously this is a rather contrived 
example, because the optimal case is if there is a split for 
every point in the dataset...

"""


import numpy as np 
from sklearn import datasets
from scipy import stats
import pandas as pd

iris = datasets.load_iris()

X = iris.data[:, 1] # take the 2nd column only
Y = iris.target

def create_bins(x, breaks):
    """ mirror of the R implementation, create bins given breaks """
    if not isinstance(breaks, list):
        raise Exception('Breaks are not a list')
        
    breaks = sorted([float("-inf"), float("inf")] + breaks)
    return pd.cut(x, breaks, labels=range(len(breaks)-1))

def get_best_classification(Y):
    """returns the number of most frequent class 
    as a percentage of total"""
    if Y.shape[0] == 0:
        return 0
    return stats.mode(Y)[1][0]/float(Y.shape[0])
    

def get_metric(splits, X, Y):
    """returns the metric related to the split, the higher the better, must 
    be between 0, 1"""
    
    if not type(splits) == list:
        splits = [splits]
    x_binned = create_bins(X, splits)
    #print(x_binned)
    
    y_metric = 0
    for lbl in list(set(x_binned)):
        cond = np.where(x_binned == lbl)
        #print(lbl)
        #print(cond)
        #print(Y[cond])
        #print("\n\n")
        y_part = len(Y[cond])
        y_metric += y_part*get_best_classification(Y[cond])
    
    weighted_metric = y_metric/len(Y)
    return weighted_metric
    #base_metric = get_best_classification(Y)
    #
    #return (weighted_metric - base_metric)/(1-base_metric)

#print(get_metric(2, X, Y))
#print(get_metric(3, X, Y))

uniq_x = list(set(X.tolist()))
map_brute = {x:get_metric(x, X, Y) for x in uniq_x}
maximum_x = max(map_brute, key=map_brute.get)  # Just use 'min' instead of 'max' for minimum.
print("best solution via brute force:", maximum_x, map_brute[maximum_x])

# use the above for MCMC via MH, the distribution would be...normal 
# around the point with variance of say 2...

lower, upper = min(X), max(X)
mu, sigma = np.mean(X), np.std(X)
truncnorm_X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

N = 10000

# x0...initialize randomly...
x = float(truncnorm_X.rvs(1)[0])
x = min(X)
all_x = [x]

for _ in range(N):
    # u ~ U(0,1)
    u = np.random.uniform(size=1)[0]
    
    # x* ~ q(x*|xi)
    #' if we used truncnorm, the detailed balance condition will be satisfied
    x_star = float(truncnorm_X.rvs(1)[0])
    
    # pi(x*)q(x|x*)    
    numerator = get_metric(x_star, X, Y) * stats.truncnorm.pdf(x=x, a=(lower - mu) / sigma, 
                                                           b=(upper - mu) / sigma, loc=x_star, scale=sigma)
    
    # pi(x)q(x*|x)
    denominator = get_metric(x, X, Y) * stats.truncnorm.pdf(x=x_star, a=(lower - mu) / sigma, 
                                                        b=(upper - mu) / sigma, loc=x, scale=sigma)
    
    if denominator == 0:
        #print("denominator is 0", x, ", ", x_star)
        all_x.append(x)
    elif u < min(1, float(numerator)/denominator):
        all_x.append(x_star)
        x = x_star
    else:
        all_x.append(x)
    
    
    all_x = all_x[-100:]
    

#print("\n\n---")
#print(all_x[:-10])
#print(get_metric(all_x[-1], X, Y))
print(all_x[-1], get_metric(all_x[-1], X, Y))  

