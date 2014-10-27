// linalg_typecodes.h
#ifndef MATH_LINALG_TYPECODES_H
#define MATH_LINALG_TYPECODES_H

// linalg_python
namespace linalg_python {

// TypeCode_to_Type
template <int TypeCode> struct TypeCode_to_Type {};
#define TYPECODE_TO_TYPE(X, Y) \
  template <> struct TypeCode_to_Type<X> \
  { \
    typedef Y type; \
  }

#ifdef NPY_INT8
TYPECODE_TO_TYPE(NPY_INT8, npy_int8);
#endif
#ifdef NPY_UINT8
TYPECODE_TO_TYPE(NPY_UINT8, npy_uint8);
#endif
#ifdef NPY_INT16
TYPECODE_TO_TYPE(NPY_INT16, npy_int16);
#endif
#ifdef NPY_UINT16
TYPECODE_TO_TYPE(NPY_UINT16, npy_uint16);
#endif
#ifdef NPY_INT32
TYPECODE_TO_TYPE(NPY_INT32, npy_int32);
#endif
#ifdef NPY_UINT32
TYPECODE_TO_TYPE(NPY_UINT32, npy_uint32);
#endif
#ifdef NPY_INT64
TYPECODE_TO_TYPE(NPY_INT64, npy_int64);
#endif
#ifdef NPY_UINT64
TYPECODE_TO_TYPE(NPY_UINT64, npy_uint64);
#endif
#ifdef NPY_INT128
TYPECODE_TO_TYPE(NPY_INT128, npy_int128);
#endif
#ifdef NPY_UINT128
TYPECODE_TO_TYPE(NPY_UINT128, npy_uint128);
#endif
#ifdef NPY_INT256
TYPECODE_TO_TYPE(NPY_INT256, npy_int256);
#endif
#ifdef NPY_UINT256
TYPECODE_TO_TYPE(NPY_UINT256, npy_uint256);
#endif
#ifdef NPY_FLOAT32
TYPECODE_TO_TYPE(NPY_FLOAT32, npy_float32);
#endif
#ifdef NPY_FLOAT64
TYPECODE_TO_TYPE(NPY_FLOAT64, npy_float64);
#endif
#ifdef NPY_FLOAT80
TYPECODE_TO_TYPE(NPY_FLOAT80, npy_float80);
#endif
#ifdef NPY_FLOAT96
TYPECODE_TO_TYPE(NPY_FLOAT96, npy_float96);
#endif
#ifdef NPY_FLOAT128
TYPECODE_TO_TYPE(NPY_FLOAT128, npy_float128);
#endif
#ifdef NPY_COMPLEX16
TYPECODE_TO_TYPE(NPY_COMPLEX16, npy_complex16);
#endif
#ifdef NPY_COMPLEX64
TYPECODE_TO_TYPE(NPY_COMPLEX64, npy_complex64);
#endif
#ifdef NPY_COMPLEX128
TYPECODE_TO_TYPE(NPY_COMPLEX128, npy_complex128);
#endif
#ifdef NPY_COMPLEX160
TYPECODE_TO_TYPE(NPY_COMPLEX160, npy_complex160);
#endif
#ifdef NPY_COMPLEX192
TYPECODE_TO_TYPE(NPY_COMPLEX192, npy_complex192);
#endif
#ifdef NPY_COMPLEX256
TYPECODE_TO_TYPE(NPY_COMPLEX256, npy_complex256);
#endif

// Type_to_TypeCode
template <typename Type> struct Type_to_TypeCode {};
#define TYPE_TO_TYPECODE(X, Y) \
  template <> struct Type_to_TypeCode<X> \
  { \
    static const int type_code = Y; \
  };

#ifdef NPY_INT8
TYPE_TO_TYPECODE(npy_int8, NPY_INT8);
#endif
#ifdef NPY_UINT8
TYPE_TO_TYPECODE(npy_uint8, NPY_UINT8);
#endif
#ifdef NPY_INT16
TYPE_TO_TYPECODE(npy_int16, NPY_INT16);
#endif
#ifdef NPY_UINT16
TYPE_TO_TYPECODE(npy_uint16, NPY_UINT16);
#endif
#ifdef NPY_INT32
TYPE_TO_TYPECODE(npy_int32, NPY_INT32);
#endif
#ifdef NPY_UINT32
TYPE_TO_TYPECODE(npy_uint32, NPY_UINT32);
#endif
#ifdef NPY_INT64
TYPE_TO_TYPECODE(npy_int64, NPY_INT64);
#endif
#ifdef NPY_UINT64
TYPE_TO_TYPECODE(npy_uint64, NPY_UINT64);
#endif
#ifdef NPY_INT128
TYPE_TO_TYPECODE(npy_int128, NPY_INT128);
#endif
#ifdef NPY_UINT128
TYPE_TO_TYPECODE(npy_uint128, NPY_UINT128);
#endif
#ifdef NPY_INT256
TYPE_TO_TYPECODE(npy_int256, NPY_INT256);
#endif
#ifdef NPY_UINT256
TYPE_TO_TYPECODE(npy_uint256, NPY_UINT256);
#endif
#ifdef NPY_FLOAT32
TYPE_TO_TYPECODE(npy_float32, NPY_FLOAT32);
#endif
#ifdef NPY_FLOAT64
TYPE_TO_TYPECODE(npy_float64, NPY_FLOAT64);
#endif
#ifdef NPY_FLOAT80
TYPE_TO_TYPECODE(npy_float80, NPY_FLOAT80);
#endif
#ifdef NPY_FLOAT96
TYPE_TO_TYPECODE(npy_float96, NPY_FLOAT96);
#endif
#ifdef NPY_FLOAT128
TYPE_TO_TYPECODE(npy_float128, NPY_FLOAT128);
#endif
#ifdef NPY_COMPLEX64
TYPE_TO_TYPECODE(npy_complex64, NPY_COMPLEX64);
#endif
#ifdef NPY_COMPLEX128
TYPE_TO_TYPECODE(npy_complex128, NPY_COMPLEX128);
#endif
#ifdef NPY_COMPLEX160
TYPE_TO_TYPECODE(npy_complex160, NPY_COMPLEX160);
#endif
#ifdef NPY_COMPLEX192
TYPE_TO_TYPECODE(npy_complex192, NPY_COMPLEX192);
#endif
#ifdef NPY_COMPLEX256
TYPE_TO_TYPECODE(npy_complex256, NPY_COMPLEX256);
#endif

} // namespace linalg_python

#endif // MATH_LINALG_TYPECODES_H
