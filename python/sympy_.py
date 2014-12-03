##########################################
# File: sympy_.py                        #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import sympy as sp
from operator import attrgetter

# sympy_polynomial_to_function
def sympy_polynomial_to_function(expr, symbs,
                                 func_name='sympy_polynomial',
                                 return_string=False):
    poly = sp.poly(sp.collect(sp.expand(expr), symbs), *symbs)
    input_names = map(attrgetter('name'), poly.gens)
    num_names = len(input_names)

    def term_string((powers, coeff)):
        indices = filter(lambda i: powers[i] != 0,
                         range(num_names))
        if indices:
            terms = map(lambda i: '(%s ** %s)' % (input_names[i],
                                                  powers[i]),
                        indices)
            return '%s * %s' % (float(coeff), ' * '.join(terms))
        else:
            return '%s' % float(coeff)

    poly_str = ' + '.join(map(term_string, poly.terms()))
    function_str = 'def %s(%s):\n    return %s' % (
        func_name, ', '.join(input_names), poly_str)

    if return_string:
        return function_str

    exec_env = {}
    exec function_str in exec_env
    return exec_env[func_name]
