from nose.tools import assert_equal

from skip_list import IndexableSkiplist
#from skip_list_ import IndexableSkiplist

def test_skip_list():
    slist = IndexableSkiplist()
    l = [i**2 for i in range(30)]
    # Test trivial insertion
    for i in l:
        slist.insert(i)
    assert_equal(list(slist), l)
    slist.insert(2)
    l.insert(2, 2)
    assert_equal(list(slist), l)
    # Test removal of elements
    slist.remove(l[-2])
    l.remove(l[-2])
    assert_equal(list(slist), l)
    l.remove(l[1])
    slist.remove(slist[1])
    assert_equal(list(slist), l)
    l.remove(l[1])
    slist.remove(slist[1])
    assert_equal(list(slist), l)
    # Test random insertion of elements
    l.append(3)
    l.sort()
    slist.insert(3)
    assert_equal(list(slist), l)


if __name__ == '__main__':
    test_skip_list()

