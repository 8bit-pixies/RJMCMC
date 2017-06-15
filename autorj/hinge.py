import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# get best split by metric??

def get_classification(y):
    """
    Calculates the misclassification
    
    returns total correct, total size, accuracy    
    """
    if len(y) == 0:
        return 0
    from collections import Counter
    c_y = Counter(y)
    best = max(c_y, key=c_y.get)
    return c_y[best]

def error_on_split(x, y, split):
    """
    parameters:
    
    x : 1d vector of a particular predictor
    y : 1d vector of the response (categorical)
    split: split point of interest
    
    returns the classification error based on this split
    
    Example:
    
    ```py
    error_on_split(list(range(10)), [0,0,0,0,0,1,1,1,1,1], -1)
    error_on_split(list(range(10)), [0,0,0,0,0,1,1,1,1,1], 4.5)
    ```
    """
    x = np.array(x)
    y = np.array(y)
    
    split1 = np.where(x <  split)
    split2 = np.where(x >= split)
    
    # check if "legal" split
    # this is an extra condition so that the model doesn't have any splits with 
    # too few in one group...
    min_group_size = min(len(split1[0]), len(split2[0]))
    if float(min_group_size)/len(y) < 0.10 :
        return 1
    
    total1 = get_classification(y[split1])
    total2 = get_classification(y[split2])
    
    return 1-(float(total1+total2)/len(y))

def best_split(x, y, psplit=error_on_split, search_min=True):
    """
    parameters:
    
    x : 1d vector of a particular predictor
    y : 1d vector of the response (categorical)
    
    return the best split based on classification error
    
    tuple: split, metric score
    
    usage:
    best_split(list(range(10)), [0,0,0,0,0,1,1,1,1,1])
    """
    x = np.array(x)
    y = np.array(y)
    
    pos_split = sorted(list(set(x)))[1:-1] 
    best_split = {split:psplit(x, y, split) for split in pos_split}
    if search_min:
        split = min(best_split, key=best_split.get) 
    else:
        split = max(best_split, key=best_split.get) 
    split1 = np.where(x <  split)
    split2 = np.where(x >=  split)
    
    return (split, best_split[split], len(split1[0]), len(split2[0]))

def best_feature_split(x, y):
    """
    parameters:
    
    x : 1d vector of a particular predictor
    y : 1d vector of the response (categorical)
    
    return the best split based on classification error
    
    usage:
    
    ```py
    import numpy as np
    from sklearn import datasets
    
    iris = datasets.load_iris()
    
    X = iris.data[:100, :]
    y = iris.target[:100]
    best_feature_split(X[:, :], y)
    ```
    """
    if x.ndim > 1:
        feature_cols = x.shape[1]
        feat_split = {col: best_split(x[:, col], y) for col in range(feature_cols)}
        rpart = min(feat_split.items(), key=lambda x: x[1][1])
        return {
            'column': rpart[0],
            'node': rpart[1][0],
            'metric': rpart[1][1],
            'split_size': (rpart[1][2],rpart[1][3])
        }
    else:
        node, val, splitl, splitr = best_split(x, y)
        return {
            'column':0,
            'node': node, 
            'metric': val,
            'split_size': (splitl, splitr)
        }


class Hinge(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    
    This class should be fixed up to handle pandas dataframes...
    
    mask: the column indices to keep
    hinge: the hinge point to calculate    
    """
    
    def _best_split(self, x, y, psplit, search_min):
        """
        parameters:

        x : 1d vector of a particular predictor
        y : 1d vector of the response (categorical)

        return the best split based on classification error

        tuple: split, metric score

        usage:
        best_split(list(range(10)), [0,0,0,0,0,1,1,1,1,1])
        """
        x = np.array(x)
        y = np.array(y)

        pos_split = sorted(list(set(x)))[1:-1] 
        best_split = {split:psplit(x, y, split) for split in pos_split}
        if search_min:
            split = min(best_split, key=best_split.get) 
        else:
            split = max(best_split, key=best_split.get) 
        split1 = np.where(x <  split)
        split2 = np.where(x >=  split)

        return (split, best_split[split], len(split1[0]), len(split2[0]))
    
    def __init__(self, mask=0, hinge=None, psplit=error_on_split, search_min=True):
        """
        psplit: is a function which determines how things should be split, 
        an example is provided in this file. 
        
        search_min: determines if the function to be search is minimimized (error) or maximized (auc)
        """
        self.mask = mask
        self.hinge = hinge
        self.psplit = psplit
        self.search_min = search_min
    
    def fit(self, x, y=None):
        """
        please fix this up
        """
        if type(self.mask) is str:
            x = np.array(x[self.mask])
        else:
            x = x[:, self.mask] 
        
        hinge_point, metric, _, _ = self._best_split(x, y, self.psplit, self.search_min)
        self.hinge = hinge_point
        return self
    
    def transform(self, x, transform="both"):
        """
        very ugly code, please fix up        
        """
        if type(self.mask) is str:
            x1 = np.array(x[self.mask])
        else:
            x1 = x[:, self.mask] 
        x1_shape = x1.shape
        pos_hinge = np.maximum.reduce([x1-self.hinge, np.zeros(x1_shape)])
        neg_hinge = np.maximum.reduce([self.hinge-x1, np.zeros(x1_shape)])
                
        if type(self.mask) is str:
            knot_point = "{0:.2f}".format(round(self.hinge,2))
            knot_point = knot_point.replace(".", "_")
            pos_name = "{}_poshinge{}".format(self.mask, knot_point)
            neg_name = "{}_neghinge{}".format(self.mask, knot_point)
            x[pos_name] = pos_hinge
            x[neg_name] = neg_hinge
            return x
        else:
            if transform == "both":
                return np.hstack([x, pos_hinge.reshape(-1, 1), neg_hinge.reshape(-1, 1)])
            elif transform == "positive":
                return np.hstack([x, pos_hinge.reshape(-1, 1)])
            else:
                return np.hstack([x, neg_hinge.reshape(-1, 1)])
        

if __name__ == "__main__":
    import numpy as np
    from sklearn import datasets

    iris = datasets.load_iris()

    X = iris.data[:100, :]
    y = iris.target[:100]

    hinge = Hinge()
    hinge.fit(X, y)
