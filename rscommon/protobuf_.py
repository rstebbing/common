##########################################
# File: protobuf_.py                     #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np

# load_array
def load_array(s, stem, n=1, delimiter='_'):
    attrs = ['%s%s%d' % (stem, delimiter, n)]
    if n == 1:
        attrs.append(stem)

    for attr in attrs:
        l = getattr(s, attr, None)
        if l is not None:
            break
    else:
        raise ValueError('unable to find attrs: %s' % str(attrs))

    A = np.asarray(l)
    if n > 1:
        A = A.reshape(-1, n)
    return A

# append_array
def append_array(s, stem, A, clear=False, delimiter='_'):
    A = np.asarray(A)
    if A.ndim == 1:
        attrs = ['%s%s1' % (stem, delimiter), stem]
        A = A[:, np.newaxis]
    elif A.ndim == 2:
        attrs = ['%s%s%d' % (stem, delimiter, A.shape[1])]
        pass
    else:
        raise ValueError('cannot handle A with ndim = %d' % A.ndim)

    for attr in attrs:
        l = getattr(s, attr, None)
        if l is not None:
            break
    else:
        raise ValueError('unable to find attrs: %s' % str(attrs))

    if clear:
        del l[:]
    a = A.ravel()
    if a.dtype == bool:
        a = (bool(a_) for a_ in a)
    for a_ in a:
        l.append(a_)
