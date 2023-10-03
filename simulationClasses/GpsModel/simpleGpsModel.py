import math
import random
from copy import deepcopy

import geopy
import numpy as np
from scipy.stats import multivariate_normal

from simulationClasses.GpsModel.gpsModel import GpsModel


def shift_position(old_lat, old_lon, shift_m_lat, shift_m_long):
    start = geopy.Point(old_lat, old_lon)
    d = geopy.distance.distance(meters=shift_m_lat)
    end = d.destination(point=start, bearing=0)

    start = end
    d = geopy.distance.distance(meters=shift_m_long)
    end = d.destination(point=start, bearing=90)

    return end[0], end[1]


"""
Final Version of the GPS-Model
"""


class SimpleGpsModel(GpsModel):
    def __init__(self, model_name, vehicle_id, vehicle_factor):
        super(SimpleGpsModel, self).__init__(model_name)

        self.vehicle_id = vehicle_id
        self.vehicle_factor = vehicle_factor

        self.errors = []
        self.d_points = []

        self.next_update_time = 0

        self.fix_latitude = None
        self.fix_longitude = None
        self.fix_error = None
        self.fix_time = None
        self.heading = None
        self.speed = None
        self.acc = None

    def update_current_fix(self, real_latitude, real_longitude, real_time):
        # check for updating the gps_fix

        # at the start or if empty
        if not self.errors:
            self.errors = self.simulate_error(n=100)

        if not self.d_points:
            for i in range(5):
                d_point = self.generate_deviation_point(self.errors[0])
                self.errors.pop(0)
                self.d_points.append(d_point)

        if not self.next_update_time:
            self.next_update_time = real_time + random.choices(list(self.gps_frequency.keys()),
                                                               weights=self.gps_frequency.values(), k=1)[0]

        if real_time > self.next_update_time:

            self.next_update_time += random.choices(list(self.gps_frequency.keys()),
                                                    weights=self.gps_frequency.values(), k=1)[0]

            smoothed_d_point = self.get_smoothed_d_point()
            self.d_points.pop(0)
            d_point = self.generate_deviation_point(self.errors[0])
            self.d_points.append(d_point)

            last_latitude = self.fix_latitude
            last_longitude = self.fix_longitude
            last_time = self.fix_time

            self.fix_latitude = real_latitude + (smoothed_d_point[0] * self.vehicle_factor)
            self.fix_longitude = real_longitude + (smoothed_d_point[1] * self.vehicle_factor)
            self.fix_time = real_time
            self.fix_error = self.errors[0]

            if last_latitude is not None:
                if "bike" in self.vehicle_id:
                    self.heading = self.calculate_heading(p1=(last_latitude, last_longitude),
                                                          p2=(self.fix_latitude, self.fix_longitude))

                    last_speed = deepcopy(self.speed)

                    self.speed = self.calculate_speed(p1=(last_latitude, last_longitude),
                                                      p2=(self.fix_latitude, self.fix_longitude),
                                                      t1=last_time, t2=self.fix_time)

                    new_speed = deepcopy(self.speed)

                    if last_speed and new_speed:
                        self.acc = self.calculate_acc(v1=last_speed, v2=new_speed, t1=last_time, t2=self.fix_time)
                    else:
                        self.acc = 0

            self.errors.pop(0)

            return True
        else:
            return False

    def calculate_heading(self, p1, p2):
        if p1 == p2:
            return None
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = (1, 0)
        heading_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        heading_deg = heading_rad * 180 / math.pi
        return heading_deg

    def calculate_speed(self, p1, p2, t1, t2):
        s = math.dist(p1, p2)
        t = abs(t2 - t1)
        if not t == 0:
            v = s / t
            return v
        else:
            return 0

    def calculate_acc(self, v1, v2, t1, t2):
        a = (v2 - v1) / (t2 - t1)
        return a

    def get_current_fix(self):
        return {"latitude": self.fix_latitude,
                "longitude": self.fix_longitude,
                "time": self.fix_time,
                "error": self.fix_error,
                "heading": self.heading,
                "speed": self.speed,
                "acc": self.acc}

    def simulate_error(self, n):
        LONGTIME = 20

        for major_minor in ["major", "minor"]:

            if major_minor == "major":
                mean_error = self.mean_error_major
                min_error = self.min_error_major
                max_error = self.max_error_major
            elif major_minor == "minor":
                mean_error = self.mean_error_minor
                min_error = self.min_error_minor
                max_error = self.max_error_minor

            sim_changes = []
            sim_error = [mean_error]

            for _ in range(n):

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

                        rand_influence = \
                        random.choices(list(self.longtime_changes_probs.keys()), weights=add_weights, k=1)[0]
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
                    if (last_value + v) < min_error or (last_value + v) > max_error:
                        weights_changes[i] = 0

                if np.random.uniform(0, 1) < self.prob_change:
                    if all(v == 0 for v in weights_changes):
                        change = random.choices(values_changes, weights=list(self.changes_probs.values()), k=1)[0]
                    else:
                        change = random.choices(values_changes, weights=weights_changes, k=1)[0]
                else:
                    change = 0

                sim_changes.append(change)
                sim_error.append(sim_error[-1] + change)

            if major_minor == "major":
                sim_major_error = sim_error
            elif major_minor == "minor":
                sim_minor_error = sim_error

        sim_orientation_error = []

        VARIANZ_IN_ORIENTATION = 10
        for _ in range(n):
            sim_orientation_error.append(np.random.normal(self.error_orientation, VARIANZ_IN_ORIENTATION))

        sim_error = list(zip(sim_orientation_error, sim_major_error, sim_minor_error))
        return sim_error

    def generate_deviation_point(self, error):
        heatmap_model = self.heatmap
        heatmap_model = (heatmap_model - np.min(heatmap_model)) / (np.max(heatmap_model) - np.min(heatmap_model))
        heatmap_size = self.heatmap_size

        xlim = (heatmap_size[0], heatmap_size[1])
        ylim = (heatmap_size[2], heatmap_size[3])
        xres = len(np.sum(heatmap_model, axis=0))
        yres = len(np.sum(heatmap_model, axis=1))

        x_bin_width = (xlim[1] - xlim[0]) / xres
        # 0.4 instead if 0.5 to avoid rounding-errors
        x_coordinates = np.arange(start=xlim[0] + 0.4 * x_bin_width, stop=xlim[1] - 0.4 * x_bin_width, step=x_bin_width)

        y_bin_width = (ylim[1] - ylim[0]) / yres
        # 0.4 instead if 0.5 to avoid rounding-errors
        y_coordinates = np.arange(start=ylim[0] + 0.4 * y_bin_width, stop=ylim[1] - 0.4 * y_bin_width, step=y_bin_width)

        # starting up left and going row by row
        coordinates_list = []
        for y in y_coordinates:
            for x in x_coordinates:
                coordinates_list.append((x, y))

        x = np.linspace(xlim[0], xlim[1], xres)
        y = np.linspace(ylim[0], ylim[1], yres)
        xx, yy = np.meshgrid(x, y)

        center = (0, 0)
        major = error[1]
        minor = error[2]
        angle = error[0]

        phi = (angle / 180) * math.pi - 90
        var_x = major ** 2 * math.cos(phi) ** 2 + minor ** 2 * math.sin(phi) ** 2
        var_y = major ** 2 * math.sin(phi) ** 2 + minor ** 2 * math.cos(phi) ** 2
        cov_xy = (major ** 2 - minor ** 2) * math.sin(phi) * math.cos(phi)

        s1 = [[var_x, cov_xy], [cov_xy, var_y]]
        k1 = multivariate_normal(mean=center, cov=s1)

        # evaluate kernels at grid points
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        zz_1 = k1.pdf(xxyy)

        # reshape
        heatmap_error = zz_1.reshape((xres, yres))
        heatmap_error = (heatmap_error - np.min(heatmap_error)) / (np.max(heatmap_error) - np.min(heatmap_error))

        # combine both heatmaps
        heatmap_combine = np.add(heatmap_error, heatmap_model)

        # select random from combined heatmap
        d_point = random.choices(coordinates_list, weights=heatmap_combine.flatten(), k=1)[0]

        return d_point

    def get_smoothed_d_point(self):
        avg_x = (self.d_points[0][0] + self.d_points[1][0] + self.d_points[2][0] + self.d_points[3][0] + self.d_points[4][0]) / 5
        avg_y = (self.d_points[0][1] + self.d_points[1][1] + self.d_points[2][1] + self.d_points[3][1] + self.d_points[4][1]) / 5

        return (avg_x, avg_y)
