import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class Interaction(BaseEstimator, TransformerMixin):
    def __init__(self, interaction=0):
        self.interaction = interaction
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X_df):
        from functools import reduce
        X = X_df.copy()
        try:
            interaction_list = list(self.interaction)
        except:
            interaction_list = [self.interaction]
        if len(interaction_list) == 1:
            self.interaction = interaction_list[0]
        if type(self.interaction) is int:
            x1 = X[:, self.interaction]
            x1_interaction = np.multiply(x1, x1)
            return np.hstack([X, x1_interaction.reshape(-1, 1)])
        elif type(self.interaction) is str:
            # when it is a pandas dataframe
            x1 = np.array(X[self.interaction])
            x1_interaction = np.multiply(x1, x1)
            #return np.hstack([X, x1_interaction.reshape(-1, 1)])
            col_name = "{}_{}".format(self.interaction, self.interaction)
            X[col_name] = x1_interaction
            return X
        else:
            interaction_list = list(self.interaction)            
            if type(interaction_list[0]) is int:
                # check that it is list[int]                
                x1_interaction = reduce(np.multiply, 
                                        [X[:, idx] for idx in interaction_list])
                return np.hstack([X, x1_interaction.reshape(-1, 1)])
            elif type(interaction_list[0]) is str:
                # check that it is str, and pull out the two (or more) columns
                x1 = np.array(X[interaction_list])
                x1_interaction = reduce(np.multiply, 
                                        [X[idx] for idx in interaction_list])
                col_name = "_".join(interaction_list)
                X[col_name] = x1_interaction
                return X
            else:
                print(interaction_list)
                print(type(interaction_list[0]))
                raise Exception("invalid interaction parameter")
        return None



if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()

    X = iris.data[:100, :]
    y = iris.target[:100]
    
    inter = Interaction()
    print(inter.fit_transform(X))
    
    inter = Interaction([0,1,2])
    print(inter.fit_transform(X))
    
    import pandas as pd
    X_pd = pd.DataFrame(X)
    X_pd.columns = ['x1', 'x2', 'x3', 'x4']
    inter = Interaction(['x1', 'x2'])
    a = inter.fit_transform(X_pd)
    print(a.head())