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
        self.errors = []
        self.ordered_points = []

        self.points_to_generate = 2000
        self.current_point = 0

    def apply_inaccuracy(self, latitude, longitude):
        # change coordinates here

        if not self.d_points:
            self.generate_points()
            self.simulate_errors()
            self.order_points()

        if self.current_point == (self.points_to_generate - 1):
            print("No order_points left. apply_inaccuracy can not be used anymore. Correct position returned.")
            return latitude, longitude

        shift_lat_m = self.ordered_points[self.current_point][0]
        shift_lon_m = self.ordered_points[self.current_point][1]
        error = self.errors[self.current_point]
        self.current_point += 1

        new_lat = latitude + (shift_lat_m * self.vehicle_factor)
        new_lon = longitude + (shift_lon_m * self.vehicle_factor)

        return new_lat, new_lon, error

    def simulate_errors(self):
        LONGTIME = 20

        sim_changes = []
        sim_error = [self.mean_error]

        for _ in range(self.points_to_generate):

            values_changes = list(self.changes_probs.keys())
            weights_changes = list(self.changes_probs.values())

            if len(sim_changes) >= LONGTIME:
                current_longtime_change = sim_changes[-LONGTIME] - sim_changes[-1]

                for i, v in enumerate(values_changes):
                    weight = weights_changes[i]

                    add_weights = []
                    for key, value in self.longtime_changes_probs.items():
                        diff = abs(current_longtime_change - key)
                        add_weights.append(diff * value)
                    add_weights = [w / len(add_weights) for w in add_weights]

                    rand_influence = random.choices(list(self.longtime_changes_probs.keys()), weights=add_weights, k=1)[0]
                    longtime_influence = 0.5 * abs(rand_influence - v)
                    if rand_influence > weight:
                        weights_changes[i] = weight + longtime_influence
                    if rand_influence < weight:
                        weights_changes[i] = weight - longtime_influence

                if min(weights_changes) < 0:
                    weights_changes = [w + abs(min(weights_changes)) for w in weights_changes]

            # make sure that the change will not bring the current value out of min-max-range
            last_value = sim_error[-1]
            for i, v in enumerate(values_changes):
                if (last_value + v) < self.min_error or (last_value + v) > self.max_error:
                    weights_changes[i] = 0

            if np.random.uniform(0, 1) < self.prob_change:
                change = random.choices(values_changes, weights=weights_changes, k=1)[0]
            else:
                change = 0

            sim_changes.append(change)
            sim_error.append(sim_error[-1] + change)

        self.errors = sim_error

    def generate_points(self):

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

        d_points = random.choices(coordinates_list, weights=heatmap.flatten(), k=self.points_to_generate)

        self.d_points = d_points

    def order_points(self):
        MAXIMAL_DISTANCE = 2
        d_points = deepcopy(self.d_points)

        ordered_points = []
        first_point = random.choice(d_points)
        ordered_points.append(first_point)
        d_points.remove(first_point)

        while len(d_points) > 0:

            # nearest neighbor jump dependend on error-change

            if len(self.errors) >= 2:
                error_change = self.errors[len(ordered_points)] - self.errors[len(ordered_points) - 1]
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
