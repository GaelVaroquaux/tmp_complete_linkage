import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances

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


def nn_chain_core(distances):
    # Here distances should be a sparse matrix
    distances = sparse.lil_matrix(distances)
    n = len(distances)
    # Remove diagonal from connectivity matrix
    distances.setdiag(np.zeros(distances.shape[0]))

    # create distance sparse structure
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(distances.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.int)
    coord_col = np.array(coord_col, dtype=np.int)
    # Data structure to hold connectivity and distances can probably be
    # something like an array of lists: O(1) access in one direction, and


    # They will be at most 2*n nodes in the hierarchical clustering
    # algorithm. We can preallocate and use arrays
    active = np.zeros(2*n, dtype=np.bool)
    active[:n] = 1
    chain = list()
    L = list()
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
        L.append((a, b, big_distances[a, b]))
        # augment the distance matrix:
        new_distances = np.maximum(big_distances[a, :this_n],
                                   big_distances[b, :this_n])
        big_distances[this_n, :this_n] = new_distances
        big_distances[:this_n, this_n] = new_distances
        active[this_n] = 1
        # remove the two nodes that just got merged
        active[a] = 0
        active[b] = 0
        big_distances[a, :this_n] = np.inf
        big_distances[b, :this_n] = np.inf
        big_distances[:this_n, a] = np.inf
        big_distances[:this_n, b] = np.inf
    return L


if __name__ == '__main__':
    N = 2000
    X = np.random.random((N, 2))
    d = euclidean_distances(X)

    L = nn_chain_core_full(d)
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

