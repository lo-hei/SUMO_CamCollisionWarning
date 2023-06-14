import math
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import matplotlib.collections as mcoll
import geopy.distance
from matplotlib import pyplot as plt


def coord_mesh_to_meter_mesh(lat, long):
    """
    coverts coordinates with deg-mesh (lat, long) to an meter-mesh
    :param lat: latitude-list
    :param long: longitude-list
    :return: x-list in meter, y-list in meter
    """
    lower_left = [min(long), min(lat)]

    x_meter = []
    y_meter = []

    for i in range(len(lat)):
        coords_1 = [long[i], lower_left[1]]
        coords_2 = [lower_left[0], lat[i]]

        d_x = geopy.distance.geodesic(lower_left, coords_2).m
        d_y = geopy.distance.geodesic(lower_left, coords_1).m

        x_meter.append(d_x)
        y_meter.append(d_y)

    return x_meter, y_meter


def closest_coord_index(input_list, coord):
    distances = []

    for x in input_list:
        d = geopy.distance.geodesic(x, coord).m
        distances.append(d)

    distances = np.asarray(distances)
    i = distances.argmin()
    return i


def closest_value_index(input_list, value, allowed_error=0.3):
    arr = np.asarray(input_list)

    i = (np.abs(arr - value)).argmin()
    if abs(input_list[i] - value) < allowed_error:
        return i
    else:
        return None


def min_distance_to_trail(coord, trail_coordinates):
    closest_p1 = None
    closest_p2 = None
    closest_dist = None

    trail_coordinates.append[coord]
    trail_lat, trail_lon = zip(*trail_coordinates)
    trail_x_meter, trail_y_meter = coord_mesh_to_meter_mesh(trail_lat, trail_lon)

    coord = [trail_x_meter[-1], trail_y_meter[-1]]
    trail_x_meter = trail_x_meter[:-1]
    trail_y_meter = trail_y_meter[:-1]

    for i in range(len(trail_x_meter) - 1):
        trail_p1 = [trail_x_meter[i], trail_y_meter[i]]
        trail_p2 = [trail_x_meter[i + 1], trail_y_meter[i + 1]]
        min_dist = minDistance(trail_p1, trail_p2, coord)

        if min_dist < closest_dist or closest_dist is None:
            closest_dist = min_dist
            closest_p1 = trail_p1
            closest_p2 = trail_p2

    return closest_dist


def minDistance(A, B, E):
    # Function to return the minimum distance
    # between a line segment AB and a point E

    # vector AB
    AB = [None, None];
    AB[0] = B[0] - A[0];
    AB[1] = B[1] - A[1];

    # vector BP
    BE = [None, None];
    BE[0] = E[0] - B[0];
    BE[1] = E[1] - B[1];

    # vector AP
    AE = [None, None];
    AE[0] = E[0] - A[0];
    AE[1] = E[1] - A[1];

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1];
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1];

    # Minimum distance from
    # point E to the line segment
    reqAns = 0;

    # Case 1
    if (AB_BE > 0):

        # Finding the magnitude
        y = E[1] - B[1];
        x = E[0] - B[0];
        reqAns = sqrt(x * x + y * y);

    # Case 2
    elif (AB_AE < 0):
        y = E[1] - A[1];
        x = E[0] - A[0];
        reqAns = sqrt(x * x + y * y);

    # Case 3
    else:

        # Finding the perpendicular distance
        x1 = AB[0];
        y1 = AB[1];
        x2 = AE[0];
        y2 = AE[1];
        mod = sqrt(x1 * x1 + y1 * y1);
        reqAns = abs(x1 * y2 - y1 * x2) / mod;

    return reqAns;


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