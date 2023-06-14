import math
import random
from copy import deepcopy

from matplotlib import pyplot as plt

from tools.gpsAnalyser.utils.helper import colorline


def order_points_neighbors(d_points, errors, plot=True, plot_xlim=[-20, 20], plot_ylim=[-20, 20]):
    SHUFFLE_SIZE = 3
    MAXIMAL_DISTANCE = 2

    ordered_points = []
    first_point = random.choice(d_points)
    ordered_points.append(first_point)
    d_points.remove(first_point)

    while len(d_points) > 0:

        # nearest neighbor jump dependend on error-change

        if len(errors) >= 2:
            error_change = errors[len(ordered_points)] - errors[len(ordered_points) - 1]
        else:
            error_change = 0

        # if error gets greater, move away from center
        # if error gets smaller, move towards the center

        last_point = ordered_points[-1]
        last_d_center = math.dist(last_point, [0, 0])
        distances = []
        distances_to_center = []

        for p in d_points:
            d_center = math.dist(p, [0, 0])
            distances_to_center.append(d_center)

            d = math.dist(last_point, p)

            if error_change > 0:
                if d_center > last_d_center:
                    distances.append(d)
                else:
                    distances.append(d + 1000)

            elif error_change < 0:
                if d_center < last_d_center:
                    distances.append(d)
                else:
                    distances.append(d + 1000)
            else:
                distances.append(d)

        min_dist = min(distances)
        current_point_i = distances.index(min_dist)
        current_point = d_points[current_point_i]

        if min_dist < MAXIMAL_DISTANCE:
            print("smaller than MAXIMAL_DISTANCE", min_dist)
            ordered_points.append(current_point)
            d_points.remove(current_point)
        else:
            print("greater than MAXIMAL_DISTANCE", min_dist)
            # find nearest neighbor in already ordered points
            distances_to_ordered = []
            for o in ordered_points:
                d_o = math.dist(current_point, o)
                distances_to_ordered.append(d_o)
            min_dist_ordered = min(distances_to_ordered)
            min_dist_i_ordered = distances_to_ordered.index(min_dist_ordered)

            # add point before or after this point
            if min_dist_i_ordered > 0:
                distance_before = math.dist(current_point, ordered_points[min_dist_i_ordered - 1])
            else:
                ordered_points.insert(0, current_point)
                d_points.remove(current_point)
                continue

            if min_dist_i_ordered < (len(ordered_points) - 2):
                distance_after = math.dist(current_point, ordered_points[min_dist_i_ordered + 1])
            else:
                ordered_points.append(current_point)
                d_points.remove(current_point)
                continue

            if distance_before < distance_after:
                # add current_point_i before ordered_points[min_dist_i_ordered]
                ordered_points.insert(min_dist_i_ordered - 1, current_point)
                d_points.remove(current_point)
            else:
                # add current_point_i after ordered_points[min_dist_i_ordered]
                ordered_points.insert(min_dist_i_ordered + 1, current_point)
                d_points.remove(current_point)

    if plot:
        plt.figure(figsize=(10, 5))

        x, y = list(zip(*ordered_points))
        plt.scatter(x, y, c=range(len(x)), cmap="viridis")

        lc = colorline(x, y, cmap='viridis', linewidth=3)
        ax = plt.gca()
        ax.add_collection(lc)

        plt.xlim(plot_xlim)
        plt.ylim(plot_ylim)
        plt.hlines(0, -50, 50, colors="black")
        plt.vlines(0, -50, 50, colors="black")
        plt.tight_layout()
        plt.show()

    return ordered_points


def order_points_neighbors_duplicates(d_points, plot=True, plot_xlim=[-20, 20], plot_ylim=[-20, 20]):
    # choose from nearest r
    r = 50

    points_order = []
    d_points_indices = list(range(len(d_points)))
    first_point = random.choice(d_points_indices)
    points_order.append(first_point)

    steps = len(d_points)

    for _ in range(steps):

        # nearest neighbor jump
        last_point_i = points_order[-1]
        d_points_without_itself = deepcopy(d_points_indices)
        d_points_without_itself.remove(last_point_i)

        distances = []
        for p in d_points_without_itself:
            d = math.dist(d_points[last_point_i], d_points[p])
            distances.append(d)

        sort = sorted(zip(d_points_without_itself, distances), key=lambda t: t[1])
        d_points_without_itself, distances = zip(*sort)
        d_points_without_itself = d_points_without_itself[:r]
        max_dist = distances[-1]
        distances = distances[:r]

        distances_inverse = [((max_dist + 1) - d) ** 2 for d in distances]
        next_point = random.choices(d_points_without_itself, weights=distances_inverse, k=1)[0]
        points_order.append(next_point)

        ordered_points = []
        for p in points_order:
            ordered_points.append(d_points[p])

    if plot:
        plt.figure(figsize=(10, 5))

        x, y = list(zip(*ordered_points))
        plt.scatter(x, y, c=range(len(x)), cmap="viridis")

        lc = colorline(x, y, cmap='viridis', linewidth=3)
        ax = plt.gca()
        ax.add_collection(lc)

        plt.xlim(plot_xlim)
        plt.ylim(plot_ylim)
        plt.hlines(0, -50, 50, colors="black")
        plt.vlines(0, -50, 50, colors="black")
        plt.tight_layout()
        plt.show()

    return ordered_points
