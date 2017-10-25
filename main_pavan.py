from __future__ import division
import numpy as np
from numpy.linalg import inv, solve
import time
from lyssa.utils import get_mmap, get_empty_mmap, split_dataset
from lyssa.utils.math import fast_dot, norm, normalize, norm_cols
from .utils import approx_error, get_class_atoms, force_mi, average_mutual_coherence
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd
from lyssa.classify import classifier
import warnings
import sys
from lyssa.utils import set_openblas_threads
from lyssa.utils import fast_dot, gen_even_batches, gen_batches
import sys
from .utils import init_dictionary, normalize, norm_cols, approx_error
from itertools import cycle
from lyssa.utils import set_openblas_threads


"""
This module implements the KSVD algorithm of "K-SVD: An Algorithm for Designing Overcomplete
Dictionaries for Sparse Representation" . To Run this Algorithm we should call the  ksvd_dict_learn function or create an instance of the ksvd_coder class
"""

def ksvd(Y, D, X, n_cycles=1, verbose=True):
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for c in range(n_cycles):
        for k in range(n_atoms):
            if verbose:
                sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
                sys.stdout.flush()
            # find all the datapoints that use the kth atom
            omega_k = X[k, :] != 0
            if not np.any(omega_k):
                unused_atoms.append(k)
                continue
            # the residual due to all the other atoms but k
            Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
            U, S, V = randomized_svd(Rk, n_components=1, n_iter=10, flip_sign=False)
            D[:, k] = U[:, 0]
            X[k, omega_k] = V[0, :] * S[0]
            # update the residual
            R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
        print ""
    return D, X, unused_atoms

"""
The below function implements the non-negative variant KSVD algorithm 
"""

def nn_ksvd(Y, D, X, n_cycles=1, verbose=True):
    # the non-negative variant
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for k in range(n_atoms):
        if verbose:
            sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
            sys.stdout.flush()
        # find all the datapoints that use the kth atom
        omega_k = X[k, :] != 0
        if not np.any(omega_k):
            unused_atoms.append(k)
            continue
        # the residual due to all the other atoms but k
        Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
        try:
            U, S, V = randomized_svd(Rk, n_components=1, n_iter=50, flip_sign=False)
        except:
            warnings.warn('SVD error')
            continue

        d = U[:, 0]
        x = V[0, :] * S[0]
        # projection to the constraint set
        d[d < 0] = 0
        x[x < 0] = 0

        dTd = np.dot(d, d)
        xTx = np.dot(x, x)
        if dTd <= np.finfo('float').eps or xTx <= np.finfo('float').eps:
            continue

        for j in range(n_cycles):
            d = np.dot(Rk, x) / np.dot(x, x)
            d[d < 0] = 0
            x = np.dot(d.T, Rk) / np.dot(d, d)
            x[x < 0] = 0

        _norm = norm(d)
        d = d / _norm
        x = x * _norm
        D[:, k] = d
        X[k, omega_k] = x
        # update the residual
        R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
    print ""
    return D, X, unused_atoms
"""
The below function implements the approximate KSVD algorithm 
"""

def approx_ksvd(Y, D, X, n_cycles=1, verbose=True):
  
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for c in range(n_cycles):
        for k in range(n_atoms):
            if verbose:
                sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
                sys.stdout.flush()
            # find all the datapoints that use the kth atom
            omega_k = X[k, :] != 0
            if not np.any(omega_k):
                # print "this atom is not used"
                unused_atoms.append(k)
                continue
            Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
            # update of D[:,k]
            D[:, k] = np.dot(Rk, X[k, omega_k])
            D[:, k] = normalize(D[:, k])
            # update of X[:,k]
            X[k, omega_k] = np.dot(Rk.T, D[:, k])
            # update the residual
            R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
        print ""

    return D, X, unused_atoms


def ksvd_dict_learn(X, n_atoms, init_dict='data', sparse_coder=None,
                    max_iter=20, non_neg=False, approx=False, eta=None,
                    n_cycles=1, n_jobs=1, mmap=False, verbose=True):
    """
    The K-SVD algorithm

    X: the data matrix of shape (n_features,n_samples)
    n_atoms: the number of atoms in the dictionary
    sparse_coder: must be an instance of the sparse_coding.sparse_encoder class
    approx: if true, invokes the approximate KSVD algorithm
    max_iter: the maximum number of iterations
    non_neg: if set to True, it uses non-negativity constraints
    n_cycles: the number of updates per atom (Dictionary Update Cycles)
    n_jobs: the number of CPU threads
    mmap: if set to True, the algorithm applies memory mapping to save memory
    """
    n_features, n_samples = X.shape
    shape = (n_atoms, n_samples)
    Z = np.zeros(shape)
    # dictionary initialization
    # track the datapoints that are not used as atoms
    unused_data = []
    if init_dict == 'data':
        from .utils import init_dictionary
        D, unused_data = init_dictionary(X, n_atoms, method=init_dict, return_unused_data=True)
    else:
        D = np.copy(init_dict)

    if mmap:
        D = get_mmap(D)
        sparse_coder.mmap = True

    print "dictionary initialized"
    max_patience = 10
    error_curr = 0
    error_prev = 0
    it = 0
    patience = 0
    approx_errors = []

    while it < max_iter and patience < max_patience:
        print "----------------------------"
        print "iteration", it
        print ""
        it_start = time.time()
        if verbose:
            t_sparse_start = time.time()
        # sparse coding
        Z = sparse_coder(X, D)
        if verbose:
            t_sparse_duration = time.time() - t_sparse_start
            print "sparse coding took", t_sparse_duration, "seconds"
            t_dict_start = time.time()

        # ksvd to learn the dictionary
        set_openblas_threads(n_jobs)
        if approx:
            D, _, unused_atoms = approx_ksvd(X, D, Z, n_cycles=n_cycles)
        elif non_neg:
            D, _, unused_atoms = nn_ksvd(X, D, Z, n_cycles=it)
        else:
            D, _, unused_atoms = ksvd(X, D, Z, n_cycles=n_cycles)
        set_openblas_threads(1)
        if verbose:
            t_dict_duration = time.time() - t_dict_start
            print "K-SVD took", t_dict_duration, "seconds"
            print ""
        if verbose:
            print "number of unused atoms:", len(unused_atoms)
        # replace the unused atoms in the dictionary
        for j in range(len(unused_atoms)):
            # no datapoint available to be used as atom
            if len(unused_data) == 0:
                break
            _idx = np.random.choice(unused_data, size=1)
            idx = _idx[0]
            D[:, unused_atoms[j]] = X[:, idx]
            D[:, unused_atoms[j]] = normalize(D[:, unused_atoms[j]])
            unused_data.remove(idx)

        if eta is not None:
            # do not force incoherence in the last iteration
            if it < max_iter - 1:
                # force Mutual Incoherence
                D, unused_data = force_mi(D, X, Z, unused_data, eta)
        if verbose:
            amc = average_mutual_coherence(D)
            print "average mutual coherence:", amc

        it_duration = time.time() - it_start
        # calculate the approximation error
        error_curr = approx_error(D, Z, X, n_jobs=2)
        approx_errors.append(error_curr)
        if verbose:
            print "error:", error_curr
            print "error difference:", (error_curr - error_prev)
            error_prev = error_curr
        print "duration:", it_duration, "seconds"
        if (it > 0) and (error_curr > 0.9 * error_prev or error_curr > error_prev):
            patience += 1
        it += 1
    print ""
    return D, Z


class ksvd_coder():
    """
    a wrapper to the ksvd_dict_learn function
    """

    def __init__(self, n_atoms=None, n_nonzero_coefs=None, sparse_coder=None, init_dict="data",
                 max_iter=None, non_neg=False, approx=True, eta=None, n_cycles=1, n_jobs=1,
                 mmap=False, verbose=True):
        self.n_atoms = n_atoms
        self.sparse_coder = sparse_coder
        self.max_iter = max_iter
        self.non_neg = non_neg
        self.approx = approx
        self.eta = eta
        self.n_jobs = n_jobs
        self.init_dict = init_dict
        self.n_cycles = n_cycles
        self.verbose = verbose
        self.mmap = mmap
        self.D = None

    def _fit(self, X):
        D, _ = ksvd_dict_learn(X, self.n_atoms, init_dict=self.init_dict,
                               sparse_coder=self.sparse_coder, max_iter=self.max_iter,
                               non_neg=self.non_neg, approx=self.approx, eta=self.eta, n_cycles=self.n_cycles,
                               n_jobs=self.n_jobs, mmap=self.mmap, verbose=self.verbose)
        self.D = D

    def __call__(self, X):
        self._fit(X)
        Z = self.sparse_coder(X, self.D)
        return Z

    def fit(self, X):
        self._fit(X)

    def encode(self, X):
        return self.sparse_coder(X, self.D)

    def print_params(self):
        pass

"""
    X: the data matrix of shape (n_features,n_samples)
    n_atoms: the number of atoms in the dictionary
    sparse_coder: must be an instance of the sparse_coding.sparse_encoder class
    batch_size: the number of datapoints in each iteration
    D_init: the initial dictionary. If None, we initialize it with randomly
            selected datapoints.
    eta: the learning rate
    mu:  the mutual coherence penalty
    n_epochs: the number of times we iterate over the dataset
    non_neg: if set to True, it uses non-negativity constraints
    n_jobs: the number of CPU threads
    mmap: if set to True, the algorithm applies memory mapping to save memory

    Note that a	large batch_size implies
    faster execution but high memory overhead, while
    a smaller batch_size implies
    slower execution but low memory overhead
    """

def online_dict_learn(X, n_atoms, sparse_coder=None, batch_size=None, A=None, B=None, D_init=None,
                      beta=None, n_epochs=1, verbose=False, n_jobs=1, non_neg=False, mmap=False):
    

    # dont monitor sparse coding
    sparse_coder.verbose = False
    n_features, n_samples = X.shape
    # initialize using the data
    if D_init is None:
        D, unused_data = init_dictionary(X, n_atoms, method='data', return_unused_data=True)
    else:
        D = D_init
    print "dictionary initialized"
    if mmap:
        D = get_mmap(D)

    batch_idx = gen_batches(n_samples, batch_size=batch_size)
    n_batches = len(batch_idx)
    n_iter = n_batches
    n_total_iter = n_epochs * n_iter
    _eps = np.finfo(float).eps

    if n_jobs > 1:
        set_openblas_threads(n_jobs)

    if A is None and B is None:
        A = np.zeros((n_atoms, n_atoms))
        B = np.zeros((n_features, n_atoms))

    if beta is None:
        # create a sequence that converges to one
        beta = np.linspace(0, 1, num=n_iter)
    else:
        beta = np.zeros(n_iter) + beta

    max_patience = 10
    error_curr = 0
    error_prev = 0
    patience = 0
    approx_errors = []
    incs = []
    for e in range(n_epochs):
        # cycle over the batches
        for i, batch in zip(range(n_iter), cycle(batch_idx)):
            X_batch = X[:, batch]
            # sparse coding step
            Z_batch = sparse_coder(X_batch, D)
            # update A and B
            A = beta[i] * A + fast_dot(Z_batch, Z_batch.T)
            B = beta[i] * B + fast_dot(X_batch, Z_batch.T)
            if verbose:
                progress = float((e * n_iter) + i) / n_total_iter
                sys.stdout.write("\r" + "dictionary learning" + "...:%3.2f%%" % (progress * 100))
                sys.stdout.flush()

            DA = fast_dot(D, A)
            # this part could also be parallelized w.r.t the atoms
            for k in xrange(n_atoms):
                D[:, k] = (1 / (A[k, k] + _eps)) * (B[:, k] - DA[:, k]) + D[:, k]
            # enforce non-negativity constraints
            if non_neg:
                D[D < 0] = 0
            D = norm_cols(D)
        # replace_unused_atoms(A,unused_data,i)

        if e < n_epochs - 1:
            if patience >= max_patience:
                return D, A, B
            print ""
            print "end of epoch {0}".format(e)
            error_curr = 0
            for i, batch in zip(range(n_iter), cycle(batch_idx)):
                X_batch = X[:, batch]
                # sparse coding step
                Z_batch = sparse_coder(X_batch, D)
                error_curr += approx_error(D, Z_batch, X_batch, n_jobs=n_jobs)
            if verbose:
                print ""
                print "error:", error_curr
                print "error difference:", (error_curr - error_prev)
                error_prev = error_curr
            if (e > 0) and (error_curr > 0.9 * error_prev or error_curr > error_prev):
                patience += 1

    if verbose:
        sys.stdout.write("\r" + "dictionary learning" + "...:%3.2f%%" % (100))
        sys.stdout.flush()
        print ""
    return D, A, B

 """
    a wrapper of the online_dict_learn function
    """
class online_dictionary_coder():
   

    def __init__(self, n_atoms=None, sparse_coder=None, batch_size=None, beta=None, D_init=None,
                 n_epochs=1, verbose=False, memory="low", mmap=False, non_neg=False, n_jobs=1):
        self.n_atoms = n_atoms
        self.sparse_coder = sparse_coder
        self.batch_size = batch_size
        self.beta = beta
        self.n_epochs = n_epochs
        self.A = None
        self.B = None
        self.D_init = D_init
        self.memory = memory
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.mmap = mmap
        self.non_neg = non_neg

    def __call__(self, X):
        self.fit(X)
        return self.encode(X)

    def fit(self, X):
        self.D, self.A, self.B = online_dict_learn(X, self.n_atoms, sparse_coder=self.sparse_coder,
                                                   batch_size=self.batch_size, A=self.A, B=self.B, D_init=self.D_init,
                                                   beta=self.beta, n_epochs=self.n_epochs, verbose=self.verbose,
                                                   n_jobs=self.n_jobs, non_neg=self.non_neg, mmap=self.mmap)

    def encode(self, X):
        Z = self.sparse_coder(X, self.D)
        return Z