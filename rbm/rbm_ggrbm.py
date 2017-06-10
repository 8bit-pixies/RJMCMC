
import time

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic
from sklearn.utils.fixes import expit             # logistic function
from sklearn.utils.validation import check_is_fitted

from rbm import *

class GaussianGaussainRBM(GaussianBernoulliRBM, TransformerMixin):
    """Gaussian Bernoulli Restricted Boltzmann Machine (G-RBM).

    A Restricted Boltzmann Machine with real visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    sigma : float, optional
        Controls the width of the parabola that adds
        a quadratic offset to the energy function
        due to real-valued visible units

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : bool, optional
        The verbosity level.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    `components_` : array-like, shape (n_components, n_features), optional
        Weight matrix, where n_features in the number of visible
        units and n_components is the number of hidden units.

    `intercept_hidden_` : array-like, shape (n_components,), optional
        Biases of the hidden units.

    `intercept_visible_` : array-like, shape (n_features,), optional
        Biases of the visible units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import GaussianBernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = GaussianBernoulliRBM(n_components=2, batch_size=3)
    >>> model.fit(X)
    GaussianBernoulliRBM(batch_size=3, learning_rate=0.1, n_components=2,
               n_iter=10, random_state=None, sigma=1, verbose=False)

    References
    ----------

    [1] Master thesis http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf)

    [2] Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images."
        Master's thesis, Department of Computer Science, University of Toronto (2009).
    """

    def __init__(self, n_components=256, learning_rate=0.1,
                 n_iter=10, batch_size=10, sigma=1,
                 random_state=None, verbose=False):
        self.sigma = sigma
        super(GaussianGaussainRBM, self).__init__(
            n_components, learning_rate,
            batch_size, n_iter,
            verbose, random_state)

    def _mean_visibles_given_hiddens(self, h, v):
        """Conditional probability derivation P(v|h).

           P(v|h) = N( Wh + bias_vis, sigma^2)

           Page 38 (http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf)
        """
        p = (np.dot(h, self.components_)) + self.intercept_visible_

        return norm.rvs(loc=p, scale=np.square(self.sigma))

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.

        Reference
        ---------

        Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Page 15
        """
        
        """
        t1 = (np.square(v - self.intercept_visible_) /
              (2 * np.square(self.sigma_))).sum(1)
        t2in = (safe_sparse_dot(v / self.sigma_, self.components_.T) +
                self.intercept_hidden_)
        t2 = np.log(1. + np.exp(t2in)).sum(axis=1)
        """
        
        t1 = (np.square(v - self.intercept_visible_) /
              (2 * np.square(self.sigma_))).sum(1)
        t2in = (safe_sparse_dot(v / self.sigma_, self.components_.T) +
                self.intercept_hidden_)
        t2 = np.log(1. + np.exp(t2in)).sum(axis=1)
        return -t1 + t2

    def reconstruct(self, v):
        """reconstruct by computing positive phase
           followed by the negative phase
        """
        h_ = self._sample_hiddens(v)
        v_ = self._mean_visibles_given_hiddens(h_, v)
        return v_

    def _sigma_gradient(self, v, h):
        """
            Computes the partial derivative with
            respect to sigma

            Page 41 (http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf)
        """
        t1 = (np.square(v - self.intercept_visible_) /
              np.power(self.sigma_, 3))

        t2 = (2 * v) / np.power(self.sigma_, 3)

        t3 = safe_sparse_dot(h, self.components_)

        return check_array(t1 - (t2 * t3))

    def _fit(self, v_pos, rng):
        """trains gaussian RBM"""
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._mean_visibles_given_hiddens(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]

        # update components
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(v_neg.T, h_neg).T
        self.components_ += lr * update

        # update intercepts
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        # update sigma
        self.sigma_ += lr * (self._sigma_gradient(v_pos, h_pos) -
                             self._sigma_gradient(v_neg, h_neg)).sum(axis=0)

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def init_params(self, n_feature, **settings):
        rng = super(GaussianBernoulliRBM, self).init_params(
            n_feature, **settings)
        self.sigma_ = np.ones(n_feature) * self.sigma
        return rng