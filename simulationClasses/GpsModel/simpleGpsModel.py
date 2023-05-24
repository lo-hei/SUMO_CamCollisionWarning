import math
import random
from copy import deepcopy

import geopy
import numpy as np

from simulationClasses.GpsModel.gpsModel import GpsModel


def shift_position(old_lat, old_lon, shift_m_lat, shift_m_long):
    start = geopy.Point(old_lat, old_lon)
    d = geopy.distance.distance(meters=shift_m_lat)
    end = d.destination(point=start, bearing=0)

    start = end
    d = geopy.distance.distance(meters=shift_m_long)
    end = d.destination(point=start, bearing=90)

    return end[0], end[1]


class SimpleGpsModel(GpsModel):
    def __init__(self, model_name, vehicle_factor):
        super(SimpleGpsModel, self).__init__(model_name)

        self.vehicle_factor = vehicle_factor

        self.d_points = []
        self.ordered_points = []

        self.points_to_generate = 1000
        self.current_point = 0

    def apply_inaccuracy(self, latitude, longitude):
        # change coordinates here

        if not self.d_points:
            self.generate_points(points_to_generate=self.points_to_generate)
            self.order_points()

        if self.current_point == (self.points_to_generate - 1):
            print("No order_points left. apply_inaccuracy can not be used anymore. Correct position returned.")
            return latitude, longitude

        shift_lat_m = self.ordered_points[self.current_point][0]
        shift_lon_m = self.ordered_points[self.current_point][1]
        self.current_point += 1

        new_lat = latitude + (shift_lat_m * self.vehicle_factor)
        new_lon = longitude + (shift_lon_m * self.vehicle_factor)

        return new_lat, new_lon

    def generate_points(self, points_to_generate):

        heatmap = self.heatmap
        heatmap_size = self.heatmap_size

        col_sums = np.sum(heatmap, axis=0)
        col_nums = len(col_sums)
        row_sums = np.sum(heatmap, axis=1)
        row_nums = len(row_sums)

        x_start = heatmap_size[0]
        x_end = heatmap_size[1]
        x_bin_width = (x_end - x_start) / col_nums
        # 0.4 instead if 0.5 to avoid rounding-errors
        x_coordinates = np.arange(start=x_start + 0.4 * x_bin_width, stop=x_end - 0.4 * x_bin_width, step=x_bin_width)

        y_start = heatmap_size[2]
        y_end = heatmap_size[3]
        y_bin_width = (y_end - y_start) / row_nums
        # 0.4 instead if 0.5 to avoid rounding-errors
        y_coordinates = np.arange(start=y_start + 0.4 * y_bin_width, stop=y_end - 0.4 * y_bin_width, step=y_bin_width)

        # starting up left and going row by row
        coordinates_list = []
        for y in y_coordinates:
            for x in x_coordinates:
                coordinates_list.append((x, y))

        d_points = random.choices(coordinates_list, weights=heatmap.flatten(), k=points_to_generate)

        self.d_points = d_points

    def order_points(self):
        MAXIMAL_DISTANCE = 2
        d_points = deepcopy(self.d_points)

        ordered_points = []
        first_point = random.choice(d_points)
        ordered_points.append(first_point)
        d_points.remove(first_point)

        while len(d_points) > 0:

            # nearest neighbor jump
            last_point = ordered_points[-1]
            distances = []
            for p in d_points:
                d = math.dist(last_point, p)
                distances.append(d)
            min_dist = min(distances)
            current_point_i = distances.index(min_dist)
            current_point = d_points[current_point_i]

            if min_dist < MAXIMAL_DISTANCE:
                ordered_points.append(current_point)
                d_points.remove(current_point)
            else:
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

        self.ordered_points = ordered_points
