from nose.tools import assert_equal

import numpy as np

from skip_list import IndexableSkipList

#def test_skip_list():
if 1:
    N = 5
    indices = np.arange(N)**2
    values = np.random.randint(1000, size=N).astype(np.float)
    # Test trivial insertion
    slist = IndexableSkipList()
    slist.multiple_insert(indices, values)
    assert_equal(list(slist.iteritems()), zip(indices, values))
    assert_equal(len(slist), N)
    # Test trivial insertion
    for i, v in zip(indices, values):
        slist[i + 1] = v - 1
        assert_equal(slist[i + 1], v - 1)

    l = list(slist.iteritems())
    # Test removal of elements
    slist.pop(l[-2][0])
    l.remove(l[-2])
    assert_equal(list(slist.iteritems()), l)
    slist.pop(l[1][0])
    l.remove(l[1])
    assert_equal(list(slist.iteritems()), l)
    slist.pop(l[1][0])
    l.remove(l[1])
    assert_equal(list(slist.iteritems()), l)
    # Test random insertion of elements
    l.append((3, 6.))
    l.sort()
    slist[3] = 6.
    assert_equal(list(slist.iteritems()), l)


if __name__ == '__main__':
    test_skip_list()

