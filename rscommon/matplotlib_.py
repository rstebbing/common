##########################################
# File: matplotlib_.py                   #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
from matplotlib import ticker

# set_axis_ticks
def set_axis_ticks(ax, x_or_y, axis_labels, flip, **props):
    axis_ticks = range(0, len(axis_labels))
    axis = getattr(ax, '%saxis' % x_or_y)
    axis.set_major_locator(ticker.FixedLocator(axis_ticks))
    axis.set_minor_locator(ticker.NullLocator())
    axis.set_major_formatter(ticker.FixedFormatter(axis_labels))
    axis.set_minor_formatter(ticker.NullFormatter())
    lim = (-0.5, len(axis_labels) - 0.5)
    if flip:
        lim = lim[::-1]
    set_lim = getattr(ax, 'set_%slim' % x_or_y)
    set_lim(*lim)
    if props:
        plt.setp(axis.get_majorticklabels(), **props)

# set_xaxis_ticks
def set_xaxis_ticks(ax, xaxis_labels, flip=False, **props):
    set_axis_ticks(ax, 'x', xaxis_labels, flip, **props)

# set_yaxis_ticks
def set_yaxis_ticks(ax, yaxis_labels, flip=False, **props):
    set_axis_ticks(ax, 'y', yaxis_labels, flip, **props)
