# argcheck.py

# Imports
import numpy as np

# check_ndarray_or_raise
def check_ndarray_or_raise(name, a, dtype, ndim, shape, *flags):
    """Raise a ValueError if the input array does not have the correct
    type, dimensions, shape, or flags.

    Parameters
    ----------
    name : string
        Name of the input array. Used in the exception error message.

    a : np.ndarray
        Input array.

    dtype : np.dtype or None
        Expected data type.

    ndim : int or None
        Expected number of dimensions.

    shape : tuple or None
        Expected shape. If provided, None can be used to indicate an unknown
        (but admissible) size along a given dimension e.g. shape=(None, 2).

    *flags : strings
        Attributes of `a.flags` that must be true.
    """
    check_type_or_raise(name, a, np.ndarray)

    if dtype is not None:
        d = np.dtype(dtype)
        if a.dtype != d:
            raise ValueError("Argument '%s' has incorrect dtype "
                             "expected '%s' but got '%s'" %
                             (name, str(d), str(a.dtype)))

    if ndim is not None and a.ndim != ndim:
        raise ValueError("Argument '%s' has incorrect number of dimensions "
                         "(expected %d, got %d)" %
                         (name, ndim, a.ndim))

    if shape is not None:
        shape = tuple(shape)
        raise_ = len(a.shape) != len(shape)
        if not raise_:
            for i, j in zip(a.shape, shape):
                if j is not None and i != j:
                    raise_ = True
                    break

        if raise_:
            raise ValueError("Argument '%s' has incorrect shape "
                             "(expected %s, got %s)" %
                             (name, shape, a.shape))

    for flag in flags:
        try:
            attr = getattr(a.flags, flag)
        except AttributeError:
            raise ValueError("Argument '%s' has no flag %s" %
                             (name, flag.upper()))

        if not getattr(a.flags, flag):
            raise ValueError("Argument '%s' requires flag %s" %
                             (name, flag.upper()))

# check_type_or_raise
def check_type_or_raise(name, object_, type_):
    "Raise a TypeError if the input `object_` is not an instance of `type_`."
    if not isinstance(object_, type_):
        raise TypeError("Argument '%s' has incorrect type "
                        "(expected %s, got %s)" %
                        (name, type_.__name__, type(object_).__name__))
