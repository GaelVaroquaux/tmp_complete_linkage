from nose.tools import assert_equal

from skip_list import IndexableSkiplist

def test_skip_list():
    slist = IndexableSkiplist()
    l = [i**2 for i in range(10)]
    # Test trivial insertion
    for i in l:
        slist.insert(i)
    assert_equal(list(slist), l)
    # Test removal of elements
    l.remove(1)
    slist.remove(1)
    assert_equal(list(slist), l)
    # Test random insertion of elements
    l.append(3.4)
    l.sort()
    slist.insert(3.4)
    assert_equal(list(slist), l)


if __name__ == '__main__':
    test_skip_list()

