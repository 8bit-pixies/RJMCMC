"""
mcmc.py

test script for MH algorithm for some rather silly example

states -> some number on the real number line. 

q(x | x') = q(x' | x) for RJMCMC?
evaluation: p(x) = some objective function (say tree splits)


"""


import numpy as np 
from sklearn import datasets
from scipy import stats

iris = datasets.load_iris()

X = iris.data[:, 1] # take the 2nd column only
Y = iris.target

def get_best_classification(Y):
    """returns the number of most frequent class 
    as a percentage of total"""
    if Y.shape[0] == 0:
        return 0
    return stats.mode(Y)[1][0]/float(Y.shape[0])
    

def get_metric(split, X, Y):
    """returns the metric related to the split, the higher the better, must 
    be between 0, 1"""
    split_part = np.where(X > split)
    true_part = len(split_part[0])
    weighted_metric = (true_part * get_best_classification(Y[X>split]) + (len(Y)-true_part)*get_best_classification(Y[X<=split]))/float(len(Y))
    return weighted_metric
    base_metric = get_best_classification(Y)
    
    return (weighted_metric - base_metric)/(1-base_metric)

#print(get_metric(2, X, Y))
#print(get_metric(3, X, Y))

uniq_x = list(set(X.tolist()))
map_brute = {x:get_metric(x, X, Y) for x in uniq_x}
maximum_x = max(map_brute, key=map_brute.get)  # Just use 'min' instead of 'max' for minimum.
print(maximum_x, map_brute[maximum_x])



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
    #' make it reversible...and from normal distribution?
    x_star = float(truncnorm_X.rvs(1)[0])
    
    # pi(x*)q(x|x*)
    
    numerator = get_metric(x_star, X, Y) * stats.truncnorm.pdf(x=x, a=(lower - mu) / sigma, 
                                                           b=(upper - mu) / sigma, loc=x_star, scale=sigma)
    #numerator = get_metric(x_star, X, Y) * truncnorm_X.cdf(x=x)
    
    
    # pi(x)q(x*|x)
    denominator = get_metric(x, X, Y) * stats.truncnorm.pdf(x=x_star, a=(lower - mu) / sigma, 
                                                        b=(upper - mu) / sigma, loc=x, scale=sigma)
    #denominator = get_metric(x, X, Y) * truncnorm_X.cdf(x=x_star)
    
    
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

