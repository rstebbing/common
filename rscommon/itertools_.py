##########################################
# File: itertools_.py                    #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
from itertools import *

# From (or adapted from) https://doc.python.org/2/library/itertools.html

# flatten
def flatten(list_of_lists):
    "Flatten one level of nesting"
    return chain.from_iterable(list_of_lists)

# pairwise
def pairwise(s, repeat=False):
    a, b = tee(s)
    first = next(b, None)
    if repeat:
        b = chain(b, [first])
    return izip(a, b)
