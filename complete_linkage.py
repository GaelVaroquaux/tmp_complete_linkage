import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from skip_list import IndexableSkipList
import sys



#@profile
def nn_chain_core_full(distances):
    n = len(distances)
    # They will be at most 2*n nodes in the hierarchical clustering
    # algorithm. We can preallocate and use arrays
    active = np.zeros(2*n, dtype=np.bool)
    active[:n] = 1
    chain = list()
    children = list()
    big_distances = np.ones((2*n, 2*n))
    big_distances[:] = np.inf
    big_distances[:n, :n] = distances
    big_distances.flat[::2*n+1] = np.inf
    for this_n in xrange(n, 2*n - 1):
        if len(chain) < 4:
            # Pick any 2 active elements to complete the chain
            # The last element is active: it just got added
            a = this_n - 1
            b = this_n - 2
            while not active[b]:
                b -= 1
            chain = [a, ]
        else:
            a = chain[-4]
            b = chain[-3]
            chain = chain[:-3]
        while True:
            distance_a = big_distances[a, :this_n]
            c = np.argmin(distance_a)
            if distance_a[c] == distance_a[b]:
                c = b
            a, b = c, a
            chain.append(a)
            if len(chain) > 2 and a == chain[-3]:
                break
        children.append((a, b, distance_a[a]))
        # augment the distance matrix:
        new_distances = np.maximum(big_distances[a, :this_n],
                                   distance_a)
        big_distances[this_n, :this_n] = new_distances
        big_distances[:this_n, this_n] = new_distances
        active[this_n] = 1
        # remove the two node that just got merged
        active[a] = 0
        active[b] = 0
        big_distances[a, :this_n] = np.inf
        big_distances[b, :this_n] = np.inf
        big_distances[:this_n, a] = np.inf
        big_distances[:this_n, b] = np.inf
    return children


#@profile
def nn_chain_core(distances):
    n = len(distances)
    distances = sparse.csr_matrix(distances)
    # They will be at most 2*n nodes in the hierarchical clustering
    # algorithm. We can preallocate and use arrays
    #active = np.zeros(2*n, dtype=np.bool)
    #active[:n] = 1
    chain = list()
    children = list()
    # XXX: should this be a list, or a dict
    distance_dict = dict()
    for index, col in enumerate(distances):
        # Need to be able to have an actual connectivity at some point:
        # col being a sparse matrix
        # Should catter for empty cols?
        # Probably not: they should never occur
        this_distance = IndexableSkipList(expected_size=2*n)
        this_distance.multiple_insert(col.indices, col.data)
        distance_dict[index] = this_distance
    #big_distances = np.ones((2*n, 2*n))
    #big_distances[:] = np.inf
    #big_distances[:n, :n] = distances
    #big_distances.flat[::2*n+1] = np.inf
    for this_n in xrange(n, 2*n - 1):
        if len(chain) < 4:
            # Pick any 2 active elements to complete the chain
            # The last element is active: it just got added
            a = this_n - 1
            b = this_n - 2
            while not b in distance_dict:
                b -= 1
            chain = [a, ]
        else:
            a = chain[-4]
            b = chain[-3]
            chain = chain[:-3]
        while True:
            distance_a = distance_dict[a]
            c, min_value = distance_a.argmin()
            # XXX: it would be great if we could avoid accessing
            # distance_a[b]
            if min_value == distance_a._get_node(b, default=np.inf):
                c = b
                # There's probably an optimization possible for the
                # next round
            a, b = c, a
            chain.append(a)
            if len(chain) > 2 and a == chain[-3]:
                break
        children.append((a, b, distance_a[a]))
        # Remove the nodes in the corresponding skip lists, and the
        # corresponding skip_lists from the distance dictionary
        del distance_dict[b]
        new_distances = distance_dict.pop(a)
        # Augment the distance matrix:
        #new_distances = np.maximum(new_distances, distance_a)
        for other_index, other_value in distance_a.iteritems():
            new_distances[other_index] = max(
                        new_distances._get_node(other_index, default=0),
                        other_value)
        distance_dict[this_n] = new_distances
        for value in distance_dict.itervalues():
            value._get_node(a, remove=1, default=0)
            value._get_node(b, remove=1, default=0)
        for index, value in new_distances.iteritems():
            distance_dict[index][this_n] = value
    return children



if __name__ == '__main__':
    N = 200
    np.random.seed(0)
    X = np.random.random((N, 2))
    d = euclidean_distances(X)

    L = nn_chain_core(d)
    a, b, height = np.array(L).T
    #order = np.argsort(height, kind='mergesort')
    #a = a[order]
    #b = b[order]
    #height = height[order]
    if 1:
        import pylab as pl
        children = np.c_[a, b].astype(np.int)
        from sklearn.cluster.hierarchical import _hc_cut, ward_tree
        labels = _hc_cut(n_clusters=4, children=children, n_leaves=N)
        pl.figure(1)
        pl.clf()
        pl.scatter(X[:, 0], X[:, 1], c=labels, cmap=pl.cm.spectral)
        pl.title('Complete linkage')
    if 1:
        from scipy.cluster import hierarchy
        children_s = hierarchy.complete(X)[:, :2].astype(np.int)
        labels_s = _hc_cut(n_clusters=4, children=children_s, n_leaves=N)
        import pylab as pl
        pl.figure(0)
        pl.clf()
        pl.scatter(X[:, 0], X[:, 1], c=labels_s, cmap=pl.cm.spectral)
        pl.title('Complete linkage (scipy)')
    if 0:
        pl.figure(2)
        pl.clf()
        children_w, _, _ = ward_tree(X)
        labels_w = _hc_cut(n_clusters=4, children=children_w, n_leaves=N)
        pl.scatter(X[:, 0], X[:, 1], c=labels_w, cmap=pl.cm.spectral)
        pl.title('Ward')
        pl.show()

