# Adapted to Cython from http://code.activestate.com/recipes/576930/
# Authors: Raymond Hettinger, Gael Varoquaux
# License: MIT

#import numpy as np
#cimport numpy as np

from libc.math cimport log, ceil
from libc.stdlib cimport free, malloc, rand, RAND_MAX
from libc.stdio cimport perror, fprintf, stderr

#DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

# original constructor: value, next, width
cdef struct Node:
    unsigned int value
    unsigned int nb_levels
    unsigned int* width
    Node* next

# XXX: need better/cheaper append?


cdef Node init_node(unsigned int value, unsigned int nb_levels):
    cdef Node node
    node.value = value
    node.nb_levels = nb_levels
    # XXX: malloc means manual free
    node.next = <Node*> malloc(nb_levels * sizeof(Node))
    node.width = <unsigned int*> malloc(nb_levels
                                        * sizeof(unsigned int))
    return node


cdef class IndexableSkiplist:
    """Sorted collection supporting O(lg n) insertion, removal,
    and lookup by rank."""
    cdef unsigned int maxlevels
    cdef unsigned int size
    cdef Node head

    def __init__(self, unsigned int expected_size=100):
        cdef Node terminator_node
        cdef Node head
        cdef unsigned int i
        terminator_node.value = 2**31
        terminator_node.nb_levels = 0
        self.size = 0
        self.maxlevels = 1 + <unsigned int> (log(expected_size)/log(2))
        head = init_node(0, self.maxlevels)
        for i in range(self.maxlevels):
            head.next[i] = terminator_node
            head.width[i] = 1
        self.head = head

    def __len__(self):
        return self.size

    def __getitem__(self, unsigned int i):
        cdef unsigned int level
        if i >= self.size:
            raise StopIteration
        node = self.head
        i += 1
        for level in range(self.maxlevels):
            # Access levels in descending order
            level = self.maxlevels - level - 1
            while node.width[level] <= i:
                i -= node.width[level]
                node = node.next[level]
        # Internally, we store values starting at 1, not 0, as 0 is our
        # head
        return node.value - 1

    def insert(self, unsigned int value):
        # find first node on each level where node.next[levels].value > value
        cdef Node* chain 
        cdef Node node
        cdef Node newnode
        cdef unsigned int* steps_at_level
        cdef unsigned int level, d, steps
        chain = <Node*> malloc(self.maxlevels * sizeof(Node))
        steps_at_level = <unsigned int*> malloc(self.maxlevels
                                                * sizeof(unsigned int))
        node = self.head
        steps_at_level[0] = 0
        # Internally, we store values starting at 1, not 0, as 0 is our
        # head
        value += 1
        for level in range(self.maxlevels):
            # Access levels in descending order
            level = self.maxlevels - level - 1
            steps_at_level[level] = 0
            while node.next[level].value <= value:
                steps_at_level[level] += node.width[level]
                node = node.next[level]
            chain[level] = node

        # insert a link to the newnode at each level
        d = min(self.maxlevels,
                1 - <int> ceil(log(rand()/<float>RAND_MAX)/log(2.)))
        newnode = init_node(value, d)
        steps = 0
        for level in range(d):
            prevnode = chain[level]
            newnode.next[level] = prevnode.next[level]
            prevnode.next[level] = newnode
            newnode.width[level] = prevnode.width[level] - steps
            prevnode.width[level] = steps + 1
            steps += steps_at_level[level]
        for level in range(d, self.maxlevels):
            chain[level].width[level] += 1
        self.size += 1

    def remove(self, int value):
        # find first node on each level where node.next[levels].value >= value
        cdef Node* chain 
        cdef Node node = self.head
        cdef unsigned int level, d
        # Internally, we store values starting at 1, not 0, as 0 is our
        # head
        value += 1
        chain = <Node*> malloc(self.maxlevels * sizeof(Node))

        # First find the node that needs to be removed
        for level in range(self.maxlevels):
            # Access levels in descending order
            level = self.maxlevels - level - 1
            while node.next[level].value < value:
                node = node.next[level]
            chain[level] = node
        if value != chain[0].next[0].value:
            raise KeyError('Not Found')

        # remove one link at each level
        d = chain[0].next[0].nb_levels
        for level in range(d):
            prevnode = chain[level]
            prevnode.width[level] += prevnode.next[level].width[level] - 1
            prevnode.next[level] = prevnode.next[level].next[level]
        for level in range(d, self.maxlevels):
            chain[level].width[level] -= 1
        self.size -= 1

    def __iter__(self):
        'Iterate over values in sorted order'
        cdef Node node = self.head.next[0]
        while node.nb_levels != 0:
            # Internally, we store values starting at 1, not 0, as 0 is our
            # head
            yield node.value - 1
            node = node.next[0]

