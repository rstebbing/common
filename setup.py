##########################################
# File: setup.py                         #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import os
import sys

from distutils.core import setup

modules = []

setup(name='rscommon',
      version='0.1',
      author='Richard Stebbing',
      author_email='richie.stebbing@gmail.com',
      license='MIT',
      url='https://github.com/rstebbing/common',
      packages=['rscommon'] + modules)
