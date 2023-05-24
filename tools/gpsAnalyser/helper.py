import math
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import matplotlib.collections as mcoll
import geopy.distance
from matplotlib import pyplot as plt


def closest_value_index(input_list, value, allowed_error=0.3):
    arr = np.asarray(input_list)
    i = (np.abs(arr - value)).argmin()
    if abs(input_list[i] - value) < allowed_error:
        return i
    else:
        return None


def distance_earth(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    d = geopy.distance.geodesic(coords_1, coords_2).m
    return d


def point_distance_shift(new_center_lat, new_center_lon, lat, lon):
    horizontal_shift = distance_earth(new_center_lat, new_center_lon, new_center_lat, lon)
    if new_center_lon < lon:
        horizontal_shift = horizontal_shift * (-1)

    vertical_shift = distance_earth(new_center_lat, new_center_lon, lat, new_center_lon)
    if new_center_lat > lat:
        vertical_shift = vertical_shift * (-1)
    return horizontal_shift, vertical_shift


def shift_position(old_lat, old_lon, shift_m_lat, shift_m_long):
    start = geopy.Point(old_lat, old_lon)
    d = geopy.distance.distance(meters=shift_m_lat)
    end = d.destination(point=start, bearing=0)

    start = end
    d = geopy.distance.distance(meters=shift_m_long)
    end = d.destination(point=start, bearing=90)

    return end[0], end[1]


def colorline(x, y, z=None, cmap='viridis', norm=plt.Normalize(0.0, 1.0),
              linewidth=2, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments