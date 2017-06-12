import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class Interaction(BaseEstimator, TransformerMixin):
    def __init__(self, interaction=0):
        self.interaction = interaction
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if type(self.interaction) is int:
            x1 = X[:, self.interaction]
            x1_interaction = np.multiply(x1, x1)
        else:
            from functools import reduce
            interaction_list = list(self.interaction)
            x1_interaction = reduce(np.multiply, 
                                    [X[:, idx] for idx in interaction_list])
        return np.hstack([X, x1_interaction.reshape(-1, 1)])

if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()

    X = iris.data[:100, :]
    y = iris.target[:100]
    
    inter = Interaction()
    print(inter.fit_transform(X))
    
    inter = Interaction([0,1,2])
    print(inter.fit_transform(X))