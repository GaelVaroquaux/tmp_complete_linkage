# Adapted to Cython from http://code.activestate.com/recipes/576930/
# Authors: Raymond Hettinger, Gael Varoquaux
# License: MIT

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport log, ceil
from libc.stdlib cimport free, malloc, rand, RAND_MAX
from libc.stdio cimport perror, fprintf, stderr

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

ctypedef Node* pNode

# A node: a container for our skip-list
cdef struct Node:
    float value
    unsigned int index
    unsigned int nb_levels
    unsigned int* width
    pNode* next

# XXX: need better/cheaper append?

cdef pNode init_node(unsigned int index, unsigned int nb_levels,
                    float value=0):
    cdef Node *node = <Node*> malloc(sizeof(Node))
    node.index = index 
    node.value = value
    node.nb_levels = nb_levels
    node.next = <pNode*> malloc(nb_levels * sizeof(pNode))
    node.width = <unsigned int*> malloc(nb_levels
                                        * sizeof(unsigned int))
    return node


cdef del_node(pNode node, int recursive=0):
    if recursive and node.nb_levels != 0:
            # We need to traverse the list only with the lowest level
            del_node(node.next[0], 1)
    free(node.next)
    free(node.width)
    free(node)


cdef class IndexableSkipList:
    """Sorted collection supporting O(lg n) insertion, removal,
       and lookup by index."""
    cdef unsigned int max_levels
    cdef unsigned int size
    cdef pNode head
    cdef pNode* chain_buffer

    @cython.cdivision(True)
    def __init__(self, unsigned int expected_size=100):
        cdef pNode head
        cdef unsigned int i
        # A tracer
        cdef pNode terminator_node
        terminator_node = init_node(2**31 - 1, 0)

        self.size = 0
        self.max_levels = 1 + <unsigned int> (log(expected_size)/log(2))
        head = init_node(0, self.max_levels, 2**31 - 1)
        for i in range(self.max_levels):
            head.next[i] = terminator_node
            head.width[i] = 1
        self.head = head
        # Initialize a buffer, so as not to have to reinitialize it for
        # each call
        self.chain_buffer = <pNode*> malloc(self.max_levels * sizeof(pNode))

    def __len__(self):
        return self.size

    #--------------------------------------------------------------------------
    # Setting elements
    @cython.cdivision(True)
    def __setitem__(self, unsigned int index, float value):
        # find first node on each level where node.next[levels].index > index
        cdef Node* node = self.head
        cdef Node* new_node
        cdef unsigned int* steps_at_level
        cdef unsigned int level, d, steps
        steps_at_level = <unsigned int*> malloc(self.max_levels
                                                * sizeof(unsigned int))
        # Internally, we store indices starting at 1, not 0, as 0 is our
        # head
        index += 1

        # First find where to insert our node
        for level in range(self.max_levels):
            # Access levels in descending order
            level = self.max_levels - level - 1
            steps_at_level[level] = 0
            while node.next[level].index <= index:
                steps_at_level[level] += node.width[level]
                node = node.next[level]
            self.chain_buffer[level] = node
        if node.index == index:
            # The index is already in our list
            node.value = value
        else:
            # insert a link to the new_node at each level
            d = min(self.max_levels,
                    1 - <int> ceil(log(rand()/<float>RAND_MAX)/log(2.)))
            new_node = init_node(index, d, value)
            steps = 0
            for level in range(d):
                prev_node = self.chain_buffer[level]
                new_node.next[level] = prev_node.next[level]
                prev_node.next[level] = new_node
                new_node.width[level] = prev_node.width[level] - steps
                prev_node.width[level] = steps + 1
                steps += steps_at_level[level]
            for level in range(d, self.max_levels):
                self.chain_buffer[level].width[level] += 1
            self.size += 1
        free(steps_at_level)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def multiple_insert(self, indices, np.ndarray[DTYPE_t, ndim=1] values):
        cdef np.ndarray[ITYPE_t, ndim=1] my_indices
        cdef unsigned int N = values.size
        cdef unsigned int i
        my_indices = np.asarray(indices).astype(np.int32)
        assert  my_indices.size == N
        # XXX: might be faster if the values were sorted
        for i in range(N):
            self.__setitem__(indices[i], values[i])

    #--------------------------------------------------------------------------
    # Accessing the elements
    def __getitem__(self, unsigned int index):
        return self._get_node(index, remove=0)

    cpdef _get_node(self, unsigned int index, unsigned int remove=0,
                  default=None):
        # find first node on each level where node.next[levels].index >= index
        cdef pNode to_delete
        cdef float value
        cdef Node* node = self.head
        cdef unsigned int level, d
        # Internally, we store indices starting at 1, not 0, as 0 is our
        # head
        index += 1

        # First find the node
        for level in range(self.max_levels):
            # Access levels in descending order
            level = self.max_levels - level - 1
            while node.next[level].index < index:
                node = node.next[level]
            self.chain_buffer[level] = node
        if index != self.chain_buffer[0].next[0].index:
            return default
        value = self.chain_buffer[0].next[0].value 

        # If we are removing, reorganize the link structure: remove one
        # link at each level
        if remove:
            d = self.chain_buffer[0].next[0].nb_levels
            to_delete = self.chain_buffer[0].next[0]
            for level in range(d):
                prev_node = self.chain_buffer[level]
                prev_node.width[level] += prev_node.next[level].width[level] - 1
                prev_node.next[level] = prev_node.next[level].next[level]
            for level in range(d, self.max_levels):
                self.chain_buffer[level].width[level] -= 1
            del_node(to_delete)
            self.size -= 1
        return value

    #--------------------------------------------------------------------------
    # Operations on non-zero elements

    def get_value(self, unsigned int i):
        ' Return the ith non-zero element'
        cdef unsigned int level
        cdef Node* node = self.head
        if i >= self.size:
            raise StopIteration
        i += 1
        for level in range(self.max_levels):
            # Access levels in descending order
            level = self.max_levels - level - 1
            while node.width[level] <= i:
                i -= node.width[level]
                node = node.next[level]
        # Internally, we store indices starting at 1, not 0, as 0 is our
        # head
        return node.index - 1, node.value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def items(self):
        'Iterate over values in sorted order'
        cdef pNode node = self.head.next[0]
        cdef np.ndarray[ITYPE_t, ndim=1] indices = np.empty(self.size,
                                                            dtype=np.int32)
        cdef np.ndarray[DTYPE_t, ndim=1] values = np.empty(self.size)
        cdef unsigned int i = 0
        while node.nb_levels != 0:
            # Internally, we store indices starting at 1, not 0, as 0 is our
            # head
            indices[i] = node.index - 1
            values[i] = node.value
            i += 1
            node = node.next[0]
        return indices, values

    def argmin(self):
        'Return the index of the smallest entry'
        cdef pNode node = self.head.next[0]
        cdef float min_value = node.value
        cdef int arg_min = node.index
        while node.nb_levels != 0:
            if node.value <= min_value:
                min_value = node.value
                arg_min = node.index
            node = node.next[0]
        # Internally, we store indices starting at 1, not 0, as 0 is our head
        # thus we need to return arg_min - 1
        return arg_min - 1, min_value


    #--------------------------------------------------------------------------
    # Operations with other SkipLists

    #def max(self, IndexableSkipList other):
    #    cdef pNode my_node = self.head.next[0]
    #    cdef pNode other_node = other.head.next[0]
    #    while my_node.nb_levels != 0 and other_node.nb_levels != 0:
    #        # XXX: could be done faster by iterating lookups starting
    #        # in the middle of the skiplist as in the __setitem__ code
    #        my_node = node.next[0]

    def __dealloc__(self):
        """ Frees the tree. This is called by Python when all the
        references to the object are gone. """
        del_node(self.head, 1)
        free(self.chain_buffer)



