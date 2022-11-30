# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core(
                               np.ndarray[np.double_t,
                                          ndim=2] distance_matrix):

    cdef np.ndarray[np.intp_t, ndim=1] node_labels
    cdef np.ndarray[np.intp_t, ndim=1] current_labels
    cdef np.ndarray[np.double_t, ndim=1] current_distances
    cdef np.ndarray[np.double_t, ndim=1] left
    cdef np.ndarray[np.double_t, ndim=1] right
    cdef np.ndarray[np.double_t, ndim=2] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t new_node_index
    cdef np.intp_t new_node
    cdef np.intp_t i

    result = np.zeros((distance_matrix.shape[0] - 1, 3))
    node_labels = np.arange(distance_matrix.shape[0], dtype=np.intp)
    current_node = 0
    current_distances = np.infty * np.ones(distance_matrix.shape[0])
    current_labels = node_labels
    for i in range(1, node_labels.shape[0]):
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)

        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i - 1, 0] = <double> current_node
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = current_distances[new_node_index]
        current_node = new_node

    return result

cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray size_arr
    cdef np.intp_t next_label
    cdef np.intp_t *parent
    cdef np.intp_t *size

    def __init__(self, N):
        self.parent_arr = -1 * np.ones(2 * N - 1, dtype=np.intp, order='C')
        self.next_label = N
        self.size_arr = np.hstack((np.ones(N, dtype=np.intp),
                                   np.zeros(N-1, dtype=np.intp)))
        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.size = (<np.intp_t *> self.size_arr.data)

    cdef void union(self, np.intp_t m, np.intp_t n):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

        return

    cdef np.intp_t fast_find(self, np.intp_t n):
        cdef np.intp_t p
        p = n
        while self.parent_arr[n] != -1:
            n = self.parent_arr[n]
        # label up to the root
        while self.parent_arr[p] != n:
            p, self.parent_arr[p] = self.parent_arr[p], n
        return n


cpdef np.ndarray[np.double_t, ndim=2] label(np.ndarray[np.double_t, ndim=2] L):

    cdef np.ndarray[np.double_t, ndim=2] result_arr
    cdef np.double_t[:, ::1] result

    cdef np.intp_t N, a, aa, b, bb, index
    cdef np.double_t delta

    result_arr = np.zeros((L.shape[0], L.shape[1] + 1))
    result = (<np.double_t[:L.shape[0], :4:1]> (
        <np.double_t *> result_arr.data))
    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):

        a = <np.intp_t> L[index, 0]
        b = <np.intp_t> L[index, 1]
        delta = L[index, 2]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.size[aa] + U.size[bb]

        U.union(aa, bb)

    return result_arr


cpdef np.ndarray[np.double_t, ndim=2] single_linkage(distance_matrix):

    cdef np.ndarray[np.double_t, ndim=2] hierarchy
    cdef np.ndarray[np.double_t, ndim=2] for_labelling

    hierarchy = mst_linkage_core(distance_matrix)
    for_labelling = hierarchy[np.argsort(hierarchy.T[2]), :]

    return label(for_labelling)
