##########################################
# File: face_array.py                    #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np

# sequence_to_raw_face_array
def sequence_to_raw_face_array(T):
    raw_face_list = []
    raw_face_list.append(len(T))
    for i, face in enumerate(T):
        raw_face_list.append(len(face))
        raw_face_list += list(face)

    return np.array(raw_face_list, dtype=np.int32)

# raw_face_array_to_sequence
def raw_face_array_to_sequence(raw_face_array):
    T = []
    i = 1
    while i < len(raw_face_array):
        n = raw_face_array[i]
        if i + n + 1 > len(raw_face_array):
            break
        T.append(raw_face_array[i + 1: i + n + 1])
        i += n + 1
    if len(T) != raw_face_array[0]:
        raise ValueError('len(T) != raw_face_array[0] (%d vs %d)' %
                         (len(T), raw_face_array[0]))
    return T
