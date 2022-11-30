import numba
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import time
from numba import jit
from numba import cuda
import cupy as cp
#from _hdbscan_linkage import (single_linkage, mst_linkage_core,label,)
import nvidia_smi
import math

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

def gpu_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Total memory:", info.total, "Free memory:", info.free, "Used memory:", info.used)
    nvidia_smi.nvmlShutdown()

def cupyPerformance():
    s = time.time()
    x_cpu = np.ones((1000,1000,1000))
    e = time.time()
    print(e - s)### CuPy and GPU
    s = time.time()
    x_gpu = cp.ones((1000,1000,1000))
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print(e - s)

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def distance_matrix(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        out[i, j] = math.sqrt(d)

# def pytorchdist(mat, mat):
#     return torch.cdist(Y, X)

def gpu_dist_matrix(mat):
    rows = mat.shape[0]
    block_dim = (32, 32)
    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))
    
    stream = cuda.stream()
    matGpu = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    outGpu = cuda.device_array((rows, rows))
    distance_matrix[grid_dim, block_dim](matGpu, outGpu)
    #out = out2.copy_to_host(stream=stream)
    gpu_memory()
    del matGpu
    gpu_memory()
    return outGpu

def mutual_reachability_gpu(distance_matrix, min_points=5, alpha=1.0):
    start_time = time.time()
    distance_matrix= cp.asarray(distance_matrix, dtype=np.float32)
    size = distance_matrix.shape[0]
    min_points = min(size - 1, min_points)
    # print(distance_matrix.shape)
    # print("Size:",np.size(distance_matrix))
    # print("Memory size:",distance_matrix.size * distance_matrix.itemsize)
    print("--- %s MRG ---" % (time.time() - start_time))
    try:
        start_time = time.time()
        core_distances = cp.partition(distance_matrix,
                                        min_points,
                                        axis=0)[min_points]
        print("--- %s CUPY Partition seconds ---" % (time.time() - start_time))
    except AttributeError:
        core_distances = np.sort(distance_matrix,
                                 axis=0)[min_points]
    gpu_memory()
    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha
    
    start_time = time.time()
    stage1 = cp.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    gpu_memory()
    result = cp.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    print("--- %s KNN Partition seconds ---" % (time.time() - start_time))
    gpu_memory()
    return result

def mutual_reachability(distance_matrix, min_points=5, alpha=1.0):
    size = distance_matrix.shape[0]
    min_points = min(size - 1, min_points)                      
    try:
        start_time = time.time()
        core_distances = np.partition(distance_matrix,
                                      min_points,
                                      axis=0)[min_points]
        print("--- %s Partition seconds ---" % (time.time() - start_time))
    except AttributeError:
        core_distances = np.sort(distance_matrix,
                                 axis=0)[min_points]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha
    start_time = time.time()
    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    print("--- %s KNN seconds ---" % (time.time() - start_time))
    return result

def _hdbscan_generic_gpu(
    X,
    min_samples=5,
    alpha=1.0,
    metric="euclidean",
    p=2,
    leaf_size=None,
    gen_min_span_tree=False,
    **kwargs
):
    start_time = time.time()
    rows=X
    if rows>20000:
        matblocks = rows.shape[0]/20000
    for i in range(matblocks):
        A=X[i*20000:(i+1)*20000-1]
        for j in range(matblocks):
            B=X[i*20000:(i+1)*20000-1]

    distance_matrix = gpu_dist_matrix(X)
    print("--- %s Pairwise GPU seconds ---" % (time.time() - start_time))
    # print(distance_matrix[0:10,0:10])
    # print(distance_matrix1[0:10,0:10])
    start_time = time.time()
    #mutual_reachability_ = mutual_reachability_gpu(distance_matrix, min_samples, alpha)
    print("--- %s Mutual Reachability seconds ---" % (time.time() - start_time))
    #For mutual reachability of (a,a) -> coredistance(a)
    #For mutual reachability of (a,b) -> max(coredist(a), coredist(b), dist(a,b))
    #mutual_reachability_ = cp.asnumpy(mutual_reachability_).astype(np.float64)
    start_time = time.time()
    #min_spanning_tree = mst_linkage_core(mutual_reachability_)
    print("--- %s MST SPANNING TREE seconds ---" % (time.time() - start_time))

def _hdbscan_generic(
    X,
    min_samples=10,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=None,
    gen_min_span_tree=False,
    **kwargs
):
    metric="euclidean"
    start_time = time.time()
    distance_matrix = pairwise_distances(X, metric=metric, **kwargs)
    print("--- %s Pairwise seconds ---" % (time.time() - start_time))

    start_time = time.time()
    mutual_reachability_ = mutual_reachability(distance_matrix, min_samples, alpha)
    print("--- %s Mutual Reachability seconds ---" % (time.time() - start_time))
    #For mutual reachability of (a,a) -> coredistance(a)
    #For mutual reachability of (a,b) -> max(coredist(a), coredist(b), dist(a,b))
    
    start_time = time.time()
    #min_spanning_tree = mst_linkage_core(mutual_reachability_)
    print("--- %s MST SPANNING TREE seconds ---" % (time.time() - start_time))

    '''
    # Warn if the MST couldn't be constructed around the missing distances
    if np.isinf(min_spanning_tree.T[2]).any():
        print(
            "The minimum spanning tree contains edge weights with value "
            "infinity. Potentially, you are missing too many distances "
            "in the initial distance matrix for the given neighborhood "
            "size.",
            UserWarning,
        )

    # mst_linkage_core does not generate a full minimal spanning tree
    # If a tree is required then we must build the edges from the information
    # returned by mst_linkage_core (i.e. just the order of points to be merged)
    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(isclose(mutual_reachability_[int(row[1])], row[2]))[0]
            candidates = np.intersect1d(
                candidates, min_spanning_tree[:index, :2].astype(int)
            )
            candidates = candidates[candidates != row[1]]
            assert len(candidates) > 0
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    return single_linkage_tree, result_min_span_tree
    '''


moons, _ = data.make_moons(n_samples=25000, noise=0.05)
blobs, _ = data.make_blobs(n_samples=25000, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])
X = test_data

start_time = time.time()
#_hdbscan_generic(X)
print("--- %s CPU Total seconds ---" % (time.time() - start_time))

start_time = time.time()
_hdbscan_generic_gpu(X)
print("--- %s GPU Total seconds ---" % (time.time() - start_time))
gpu_memory()












'''
def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix."""
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)

def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel."""

    if Y is None:
        Y = X
    X, Y, dtype = _return_float_dtype(X, Y)

    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(_dist_wrapper)
    ret = np.empty((X.shape[0], Y.shape[0]), dtype=dtype, order="F")
    Parallel(backend="threading", n_jobs=n_jobs)(
        fd(func, ret, s, X, Y[s], **kwds)
        for s in gen_even_slices(_num_samples(Y), effective_n_jobs(n_jobs))
    )

    if (X is Y or Y is None) and func is euclidean_distances:
        # zeroing diagonal for euclidean norm.
        # TODO: do it also for other norms.
        np.fill_diagonal(ret, 0)

    return ret

def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    """Computational part of euclidean_distances

    Assumes inputs are already checked.

    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    TODO: use a float64 accumulator in row_norms to avoid the latter.
    """
    if X_norm_squared is not None:
        if X_norm_squared.dtype == np.float32:
            XX = None
        else:
            XX = X_norm_squared.reshape(-1, 1)
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            if Y_norm_squared.dtype == np.float32:
                YY = None
            else:
                YY = Y_norm_squared.reshape(1, -1)
        elif Y.dtype == np.float32:
            YY = None
        else:
            YY = row_norms(Y, squared=True)[np.newaxis, :]

    if X.dtype == np.float32:
        # To minimize precision issues with float32, we compute the distance
        # matrix on chunks of X and Y upcast to float64
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        # if dtype is already float64, no need to chunk and upcast
        distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.

    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum("ij,ij->i", X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms
    '''