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

    dtype : np.dtype
        Expected data type.

    ndim : int
        Expected number of dimensions.

    shape : tuple or None
        Expected shape.

    *flags : strings
        Attributes of `a.flags` that must be true.
    """
    check_type_or_raise(name, a, np.ndarray)

    d = np.dtype(dtype)
    if a.dtype != d:
        raise ValueError("Argument '%s' has incorrect dtype "
                         "expected '%s' but got '%s'" %
                         (name, str(d), str(a.dtype)))

    if a.ndim != ndim:
        raise ValueError("Argument '%s' has incorrect number of dimensions "
                         "(expected %d, got %d)" %
                         (name, ndim, a.ndim))

    if shape is not None:
        if a.shape != shape:
            raise ValueError("Argument '%s' has incorrect shape "
                             "(expected %s, got %s)" %
                             (name, shape, a.shape))

    for flag in flags:
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


