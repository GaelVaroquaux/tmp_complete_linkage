import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from skip_list import IndexableSkipList


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
    n = distances.shape[0]
    # XXX: Being smart with iterators is probably creating more lines than it
    # is saving
    if sparse.issparse(distances):
        distance_iter = ((d.indices, d.data)
                         for d in sparse.csr_matrix(distances))
    else:
        def distance_iter_():
            for i, col in enumerate(distances):
                indices = np.arange(n)
                indices = indices[indices != i]
                yield indices, col[indices]
        distance_iter = distance_iter_()
    # They will be at most 2*n nodes in the hierarchical clustering
    # algorithm. We can preallocate and use arrays
    active = np.zeros(2*n, dtype=np.bool)
    active[:n] = 1
    chain = list()
    children = list()
    # XXX: should this be a list, or a dict
    distance_dict = np.empty((2*n, ), dtype=object)
    for index, (indices, data) in enumerate(distance_iter):
        # Need to be able to have an actual connectivity at some point:
        # col being a sparse matrix
        # Should catter for empty cols?
        # Probably not: they should never occur
        this_distance = IndexableSkipList(expected_size=2*n)
        this_distance.multiple_insert(indices, data)
        distance_dict[index] = this_distance
    print 'Distance matrix ready'

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
            distance_a = distance_dict[a]
            c, min_value = distance_a.argmin()
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
        new_distances = distance_dict[a]
        distance_a = distance_dict[b]
        distance_dict[a] = None
        active[a] = False
        distance_dict[b] = None
        active[b] = False
        # Augment the distance matrix:
        #new_distances = np.maximum(new_distances, distance_a)
        indices, values = distance_a.items()
        for other_index, other_value in zip(indices, values):
            new_distances[other_index] = max(
                        new_distances._get_node(other_index, default=0),
                        other_value)
        distance_dict[this_n] = new_distances
        active[this_n] = True
        for distance_list in distance_dict[active]:
            distance_list._get_node(a, remove=1, default=0)
            distance_list._get_node(b, remove=1, default=0)
        indices, values = new_distances.items()
        for distance_list, value in zip(distance_dict[indices], values):
            distance_list[this_n] = value
    return children


def complete_linkage(X, connectivity=None, n_clusters=4):
    from sklearn.cluster.hierarchical import _hc_cut
    if connectivity is None:
        d = euclidean_distances(X)
    else:
        d_ = X[connectivity.row]
        d_ -= X[connectivity.col]
        d_ **= 2
        d_ = d_.sum(axis=-1)
        # XXX: not necessary: complete_linkage is invariant by increasing
        # function
        d_ = np.sqrt(d_)
        d = connectivity.copy()
        d.data = d_
    L = nn_chain_core(d)
    a, b, height = np.array(L).T
    children = np.c_[a, b].astype(np.int)
    labels = _hc_cut(n_clusters=n_clusters, children=children,
                     n_leaves=len(X))
    return labels


if __name__ == '__main__':
    N = 1000
    np.random.seed(0)
    X = np.random.random((N, 2))
    d = euclidean_distances(X)

    L = nn_chain_core(X)
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

