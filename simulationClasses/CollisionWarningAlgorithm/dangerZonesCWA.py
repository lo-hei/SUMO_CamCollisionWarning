import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from scipy.stats import norm

from simulationClasses.utils import helper
import simulationClasses.CollisionWarningAlgorithm.collisionWarningAlgorithm as cwa


def get_ellipse_points(x, y, error_ellipse):
    ellipse_points = [(x - error_ellipse[2], y), (x, y + error_ellipse[1]),
                      (x + error_ellipse[2], y), (x, y - error_ellipse[1])]

    points = []

    for p in ellipse_points:
        angle = (error_ellipse[0] / 180) * math.pi

        x_diff = p[0] - x
        y_diff = p[1] - y

        x_rotated = x + (x_diff * math.cos(angle) - y_diff * math.sin(angle))
        y_rotated = y + (x_diff * math.sin(angle) + y_diff * math.cos(angle))

        points.append((x_rotated, y_rotated))

    return points


def get_interpolated_value(value_a, list_a, list_b):
    # returns the interpolated value_b from list_b
    if not list_a[0] <= value_a <= list_a[-1]:
        return None

    higher_than_a = len([d for d in list_a if d >= value_a])

    lower_b = list_b[len(list_a) - higher_than_a]
    higher_b = list_b[len(list_a) - higher_than_a + 1]

    lower_a = list_a[len(list_a) - higher_than_a]
    higher_a = list_a[len(list_a) - higher_than_a + 1]

    if type(list_b[0]) == list or type(list_b[0]) == tuple:
        # coordinates tuple
        lower_higher = (value_a - lower_a) / (higher_a - lower_a)
        if len(list_b[0]) == 2:
            value_b_x = (1 - lower_higher) * lower_b[0] + lower_higher * higher_b[0]
            value_b_y = (1 - lower_higher) * lower_b[1] + lower_higher * higher_b[1]
            value_b = (value_b_x, value_b_y)
    else:
        # no points, just normal values like distance or time
        lower_higher = (value_a - lower_a) / (higher_a - lower_a)
        value_b = (1 - lower_higher) * lower_b + lower_higher * higher_b
    return value_b


def bounding_box(points):
    # Needs to be updates to create Bounding-Box which is not always parallel to axes
    # points always starting with min(x), min(y) and then clockwise other points
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (min(x_coordinates), max(y_coordinates)),
            (max(x_coordinates), max(y_coordinates)), (max(x_coordinates), min(y_coordinates))]


def calculate_segment_intersection(segment_1, segment_2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # Return true if line segments AB and CD intersect
    def intersect(a, b, c, d):
        return not (ccw(a, c, d) == ccw(b, c, d)) and (not (ccw(a, b, c) == ccw(a, b, d)))

    if not intersect(segment_1[0], segment_1[1], segment_2[0], segment_2[1]):
        return None

    xdiff = (segment_1[0][0] - segment_1[1][0], segment_2[0][0] - segment_2[1][0])
    ydiff = (segment_1[0][1] - segment_1[1][1], segment_2[0][1] - segment_2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    dd = (det(*segment_1), det(*segment_2))
    x = det(dd, xdiff) / div
    y = det(dd, ydiff) / div
    return x, y


class DangerZone:

    def __init__(self, p_1, p_2, v_1, v_2, poc, error_ellipse_1, error_ellipse_2):
        # 1 = own, 2 = other
        self.current_pos_1 = p_1
        self.current_pos_2 = p_2
        self.poc = poc

        self.time_to_poc_1 = None
        self.time_to_poc_2 = None

        self.calculate_time_to_pos(v_1, v_2)

        self.current_error_bb_1 = None
        self.current_error_bb_2 = None

        self.zone_vertices = None
        self.zone_matrix_resolution_m = 0.5
        self.zone_matrix = None
        self.create_danger_zone_from_error_ellipses(error_ellipse_1, error_ellipse_2)

        self.toc = None
        self.danger_value = 0

    def update(self, v_1, v_2, error_ellipse_1, error_ellipse_2):
        self.calculate_time_to_pos(v_1, v_2)
        self.create_danger_zone_from_error_ellipses(error_ellipse_1, error_ellipse_2)

    def set_danger_value(self):
        self.danger_value = np.mean(self.zone_matrix)

    def calculate_time_to_pos(self, v_1, v_2):
        distance_to_poc_1 = math.dist(self.current_pos_1, self.poc)
        distance_to_poc_2 = math.dist(self.current_pos_2, self.poc)

        self.time_to_poc_1 = distance_to_poc_1 / v_1
        self.time_to_poc_2 = distance_to_poc_2 / v_2

    def create_danger_zone_from_error_ellipses(self, error_ellipse_1, error_ellipse_2):
        # origin in SUMO is in lower left corner

        def get_ellipse_bb(x, y, angle_deg, major, minor):
            if not angle_deg == 0:
                t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
                [min_x, max_x] = sorted([x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) -
                                         minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t + np.pi, t)])
                t = np.arctan(minor / 2 * 1. / np.tan(np.radians(angle_deg)) / (major / 2))
                [min_y, max_y] = sorted([y + minor / 2 * np.sin(t) * np.cos(np.radians(angle_deg)) +
                                         major / 2 * np.cos(t) * np.sin(np.radians(angle_deg)) for t in (t + np.pi, t)])

                points_bb = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
            else:
                points_bb = [(x - minor, y - major), (x - minor, y + major),
                             (x + minor, y + major), (x + minor, y - major)]
            return points_bb

        def rotate_bb(x, y, bb, angle_deg):
            rotated_bb = []
            for p in bb:
                angle = (angle_deg / 180) * math.pi

                x_diff = p[0] - x
                y_diff = p[1] - y

                x_rotated = x + (x_diff * math.cos(angle) - y_diff * math.sin(angle))
                y_rotated = y + (x_diff * math.sin(angle) + y_diff * math.cos(angle))

                rotated_bb.append((x_rotated, y_rotated))

            return rotated_bb

        # calculate bounding-box for ellipse with center (x, y) without rotation
        bb_1 = get_ellipse_bb(self.current_pos_1[0], self.current_pos_1[1], 0, error_ellipse_1[1], error_ellipse_1[2])
        bb_2 = get_ellipse_bb(self.current_pos_2[0], self.current_pos_2[1], 0, error_ellipse_2[1], error_ellipse_2[2])

        # rotate bounding-box according to error-orientation
        self.current_error_bb_1 = rotate_bb(self.current_pos_1[0], self.current_pos_1[1], bb_1, error_ellipse_1[0])
        self.current_error_bb_2 = rotate_bb(self.current_pos_2[0], self.current_pos_2[1], bb_2, error_ellipse_2[0])

        points = []
        movement_vector_1 = (self.poc[0] - self.current_pos_1[0], self.poc[1] - self.current_pos_1[1])
        movement_vector_2 = (self.poc[0] - self.current_pos_2[0], self.poc[1] - self.current_pos_2[1])

        if math.dist(self.current_pos_1, self.current_pos_1) > (error_ellipse_2[1] + error_ellipse_1[1]):
            points = get_ellipse_points(self.current_pos_1, error_ellipse_1) + \
                     get_ellipse_points(self.current_pos_2, error_ellipse_2)
            self.zone_vertices = bounding_box(points)
        else:
            for error_bb_1_point in self.current_error_bb_1:
                for error_bb_2_point in self.current_error_bb_2:
                    segment_1_p2 = (
                        error_bb_1_point[0] + 2 * movement_vector_1[0], error_bb_1_point[1] + 2 * movement_vector_1[1])
                    segment_1 = [error_bb_1_point, segment_1_p2]

                    segment_2_p2 = (
                        error_bb_2_point[0] + 2 * movement_vector_2[0], error_bb_2_point[1] + 2 * movement_vector_2[1])
                    segment_2 = [error_bb_2_point, segment_2_p2]

                    p = calculate_segment_intersection(segment_1=segment_1, segment_2=segment_2)
                    if p is not None:
                        points.append(p)

            # if no points, do not update the points and use the old ones for further calculations
            if points:
                self.zone_vertices = bounding_box(points)

        width = math.floor(math.dist(self.zone_vertices[0], self.zone_vertices[3]) / self.zone_matrix_resolution_m)
        height = math.floor(math.dist(self.zone_vertices[0], self.zone_vertices[1]) / self.zone_matrix_resolution_m)
        self.zone_matrix = np.zeros(shape=(width, height))


class DangerZonesCWA(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike):
        super(DangerZonesCWA, self).__init__(bike=bike)

        self.plot_after_second = 0.5
        self.last_plot_time = 0

        # {vehicle_id: CAM, v_id: cam, ... }
        self.last_cams_from_vehicles = {}
        # {vehicle_id: (extrap_lat_own, extrap_lon_own, extrap_times_own, extrap_dist_own) ... }
        self.extrapolated_vehicle_paths = {}

        self.distance_to_attention = 500
        self.side_angle_to_attention = 90

        self.reaktion_time_bike = 1
        self.reaktion_time_car = 1
        self.emergency_brake_bike = 6
        self.emergency_brake_car = 7.5
        self.normal_brake_bike = 3
        self.normal_brake_car = 3

        self.extrapolate_time = 30
        self.extrapolate_timestep = 0.5

        self.danger_range_resolution_m = 0.1
        self.danger_range_decrease_by_1s = 5
        self.warning_interval = [40, 100]
        self.collision_interval = [100, 200]

        self.danger_zones = {}  # {vehicle_id: DangerZone, v_id: dZ, ... }
        self.danger_zones_history = []

    def extrapolate_position(self, lon, lat, v, head, acc, time):
        times_to_extrapolate = np.arange(0, self.extrapolate_time, self.extrapolate_timestep)
        extrapolated_longitude = []
        extrapolated_latitude = []
        extrapolated_times = []
        extrapolated_dist = []

        angle = (head / 180) * math.pi

        for t in times_to_extrapolate:
            d = (v * t) + (0.5 * acc * (t ** 2))

            extrapolated_longitude.append(lon + (d * math.sin(angle)))
            extrapolated_latitude.append(lat + (d * math.cos(angle)))
            extrapolated_times.append(time + t)
            extrapolated_dist.append(d)

        return extrapolated_latitude, extrapolated_longitude, extrapolated_times, extrapolated_dist

    def check(self):
        """
        Checking danger zone for every vehicle.
        Using simple extrapolation of current position from last CAM with speed, acceleration and heading
        :return:
        """
        for vehicle_id_other, cams in self.bike.received_cams.items():
            if not cams[-1]:
                continue

            # check if CAM message is valid
            if cams[-1].gps_time is None:
                continue

            # check if there is a new CAM, otherwise multiple steps (extrapolation and Danger-Zone-Calculation)
            # can be skipped
            if (vehicle_id_other not in self.last_cams_from_vehicles.keys()) or \
                    (not cams[-1] == self.last_cams_from_vehicles[vehicle_id_other]):
                new_cam = True
                self.last_cams_from_vehicles[vehicle_id_other] = cams[-1]
            else:
                new_cam = False

            # if distance between vehicles is too big, skip
            if math.dist((self.bike.gps_longitude, self.bike.gps_latitude),
                         (cams[-1].longitude, cams[-1].latitude)) > self.distance_to_attention:
                continue

            # if heading is wrong between vehicles, skip
            angle_interval_right = ((self.bike.heading - 90 - 0.5 * self.side_angle_to_attention) % 360,
                                    (self.bike.heading - 90 + 0.5 * self.side_angle_to_attention) % 360)
            angle_interval_left = ((self.bike.heading + 90 - 0.5 * self.side_angle_to_attention) % 360,
                                   (self.bike.heading + 90 + 0.5 * self.side_angle_to_attention) % 360)
            if not ((angle_interval_right[0] < cams[-1].heading < angle_interval_right[1]) or
                    (angle_interval_left[0] < cams[-1].heading < angle_interval_left[1])):
                continue

            # own position has to be calculated and extrapolated in every step
            extrap_lat_own, extrap_lon_own, extrap_times_own, extrap_dist_own = \
                self.extrapolate_position(lon=self.bike.gps_longitude,
                                          lat=self.bike.gps_latitude,
                                          v=self.bike.speed,
                                          head=self.bike.heading,
                                          acc=self.bike.longitudinal_acceleration,
                                          time=self.bike.gps_time)

            if new_cam:
                extrap_lat_other, extrap_lon_other, extrap_times_other, extrap_dist_other = \
                    self.extrapolate_position(lon=self.last_cams_from_vehicles[vehicle_id_other].longitude,
                                              lat=self.last_cams_from_vehicles[vehicle_id_other].latitude,
                                              v=self.last_cams_from_vehicles[vehicle_id_other].speed,
                                              head=self.last_cams_from_vehicles[vehicle_id_other].heading,
                                              acc=self.last_cams_from_vehicles[
                                                  vehicle_id_other].longitudinal_acceleration,
                                              time=self.last_cams_from_vehicles[vehicle_id_other].gps_time)
                self.extrapolated_vehicle_paths[vehicle_id_other] = (extrap_lat_other, extrap_lon_other,
                                                                     extrap_times_other, extrap_dist_other)
            else:
                extrap_lat_other, extrap_lon_other, extrap_times_other, extrap_dist_other = \
                    self.extrapolated_vehicle_paths[vehicle_id_other]

            # calculate poc (Point of Collision) -- finding intersection on polylines
            poc = None

            for ow in range(len(extrap_lon_own) - 1):
                segment_own = [(extrap_lon_own[ow], extrap_lat_own[ow]),
                               (extrap_lon_own[ow + 1], extrap_lat_own[ow + 1])]

                for ot in range(len(extrap_lon_other) - 1):
                    segment_other = [(extrap_lon_other[ot], extrap_lat_other[ot]),
                                     (extrap_lon_other[ot + 1], extrap_lat_other[ot + 1])]

                    # check if linesegments intersect
                    poc = calculate_segment_intersection(segment_other, segment_own)
                    if poc is not None:
                        break

                if poc is not None:
                    break

            # if trajectories do not intersect at all - no danger for both vehicles
            if poc is None:
                continue

            current_pos_own = get_interpolated_value(value_a=self.bike.simulation_manager.time,
                                                     list_a=extrap_times_own,
                                                     list_b=list(zip(extrap_lon_own, extrap_lat_own)))
            current_pos_other = get_interpolated_value(value_a=self.bike.simulation_manager.time,
                                                       list_a=extrap_times_other,
                                                       list_b=list(zip(extrap_lon_other, extrap_lat_other)))

            # create dangerZone with poc and the error-ellipses
            error_ellipse_own = self.bike.gps_error
            error_ellipse_other = [self.last_cams_from_vehicles[vehicle_id_other].semi_major_orientation,
                                   self.last_cams_from_vehicles[vehicle_id_other].semi_major_confidence,
                                   self.last_cams_from_vehicles[vehicle_id_other].semi_minor_confidence]

            if not vehicle_id_other in self.danger_zones.keys():
                danger_zone = DangerZone(p_1=current_pos_own, p_2=current_pos_other, v_1=self.bike.speed,
                                         v_2=self.last_cams_from_vehicles[vehicle_id_other].speed,
                                         poc=poc, error_ellipse_1=error_ellipse_own,
                                         error_ellipse_2=error_ellipse_other)
                self.danger_zones[vehicle_id_other] = danger_zone
            else:
                self.danger_zones[vehicle_id_other].update(v_1=self.bike.speed,
                                                           v_2=self.last_cams_from_vehicles[vehicle_id_other].speed,
                                                           error_ellipse_1=error_ellipse_own,
                                                           error_ellipse_2=error_ellipse_other)
                danger_zone = self.danger_zones[vehicle_id_other]

            self.calculate_danger_zone_matrix(danger_zone, current_pos_own, self.bike.speed,
                                              self.bike.longitudinal_acceleration,
                                              self.bike.vehicle_id, extrap_dist_own, extrap_times_own)
            self.calculate_danger_zone_matrix(danger_zone, current_pos_other,
                                              self.last_cams_from_vehicles[vehicle_id_other].speed,
                                              self.last_cams_from_vehicles[vehicle_id_other].longitudinal_acceleration,
                                              vehicle_id_other, extrap_dist_other, extrap_times_other)

            danger_zone.set_danger_value()
            print("danger_value", danger_zone.danger_value)

            if self.plot_after_second:
                if (self.bike.simulation_manager.time - self.last_plot_time) > self.plot_after_second:
                    print("Plotting CWA for time", self.bike.simulation_manager.time)
                    print("poc", poc)
                    self.plot_cwa(danger_zone,
                                  zip(extrap_lon_own, extrap_lat_own),
                                  zip(extrap_lon_other, extrap_lat_other))
                    self.last_plot_time = self.bike.simulation_manager.time

    def calculate_danger_zone_matrix(self, danger_zone, current_pos, speed, acc, vehicle_id, extrap_dist, extrap_times):
        # filling matrix with values from OWN

        # get direction from which the vehicle is approaching by finding the two nearest edges of the bounding-box
        distances_to_edges = []
        for i in range(len(danger_zone.zone_vertices)):
            d1 = math.dist(current_pos, danger_zone.zone_vertices[i])
            if i == len(danger_zone.zone_vertices) - 1:
                next_i = -1
            else:
                next_i = i + 1
            d2 = math.dist(current_pos, danger_zone.zone_vertices[next_i])
            avg_d = (d1 + d2) / 2
            distances_to_edges.append(avg_d)

        # filling matrix
        num_rows, num_cols = danger_zone.zone_matrix.shape
        nearest_edge = np.argmin(np.array(distances_to_edges))

        if nearest_edge == 0:
            # approaching from left
            nearest_edge_1 = 0
            nearest_edge_2 = 1
            vector_edge_1 = 1
            vector_edge_2 = 2
            first_rows = False
        elif nearest_edge == 1:
            # approaching from above
            nearest_edge_1 = 1
            nearest_edge_2 = 2
            vector_edge_1 = 1
            vector_edge_2 = 0
            first_rows = True
        elif nearest_edge == 2:
            # approaching from right
            nearest_edge_1 = 2
            nearest_edge_2 = 3
            vector_edge_1 = 2
            vector_edge_2 = 1
            first_rows = False
        elif nearest_edge == 3:
            # approaching from lower
            nearest_edge_1 = 0
            nearest_edge_2 = 3
            vector_edge_1 = 0
            vector_edge_2 = 1
            first_rows = True

        length_of_edge = math.dist(danger_zone.zone_vertices[nearest_edge_1],
                                   danger_zone.zone_vertices[nearest_edge_2])

        if first_rows:
            num_it = num_rows
        else:
            num_it = num_cols

        for i in range(num_it):

            vector_1_2 = (danger_zone.zone_vertices[vector_edge_2][0] - danger_zone.zone_vertices[vector_edge_1][0],
                          danger_zone.zone_vertices[vector_edge_2][1] - danger_zone.zone_vertices[vector_edge_1][1])
            edge_of_col_1 = (danger_zone.zone_vertices[nearest_edge_2][0] + (i / num_it) * vector_1_2[0],
                             danger_zone.zone_vertices[nearest_edge_2][1] + (i / num_it) * vector_1_2[1])

            edge_of_col_2 = (danger_zone.zone_vertices[nearest_edge_1][0] + (i / num_it) * vector_1_2[0],
                             danger_zone.zone_vertices[nearest_edge_1][1] + (i / num_it) * vector_1_2[1])

            middle_edge = ((edge_of_col_2[0] + edge_of_col_1[0]) / 2, (edge_of_col_2[1] + edge_of_col_1[1]) / 2)
            dist_edge = math.dist(current_pos, middle_edge)

            danger_value = self.calculate_danger_value(dist=dist_edge, vehicle_id=vehicle_id,
                                                       speed=speed, acc=acc, extrap_dist=extrap_dist,
                                                       extrap_times=extrap_times)

            if first_rows:
                num_2nd_it = num_cols
            else:
                num_2nd_it = num_rows

            for j in range(num_2nd_it):
                lower_limit = (length_of_edge / num_rows) * j
                upper_limit = (length_of_edge / num_rows) * (j + 1)
                cdf_lower_limit = norm(loc=length_of_edge / 2, scale=1).cdf(lower_limit)
                cdf_upper_limit = norm(loc=length_of_edge / 2, scale=1).cdf(upper_limit)

                norm_value = cdf_upper_limit - cdf_lower_limit
                if first_rows:
                    # adding one centimeter to avoid division by zero
                    danger_zone.zone_matrix[i][j] += (norm_value * danger_value) * \
                                                     1 / (
                                                                 abs(danger_zone.time_to_poc_1 - danger_zone.time_to_poc_2) + 0.01)
                else:
                    danger_zone.zone_matrix[j][i] += (norm_value * danger_value) * \
                                                     1 / (
                                                                 abs(danger_zone.time_to_poc_1 - danger_zone.time_to_poc_2) + 0.01)

    def calculate_danger_value(self, dist, vehicle_id, speed, acc, extrap_dist, extrap_times):
        if dist > extrap_dist[-1]:
            return 0

        emergency_brake_distance = self.get_emergency_brake_distance(speed, acc, vehicle_id)
        normal_brake_distance = self.get_normal_brake_distance(speed, acc, vehicle_id)

        time_of_normal_brake = get_interpolated_value(normal_brake_distance, extrap_dist, extrap_times)
        time_to_dist = get_interpolated_value(dist, extrap_dist, extrap_times)

        if dist < emergency_brake_distance:
            return 100

        if dist < normal_brake_distance:
            danger_value = 50 + 50 * (1 - (dist - emergency_brake_distance) / (
                    normal_brake_distance - emergency_brake_distance))
            return danger_value

        time_after_normal_break = time_to_dist - time_of_normal_brake
        danger_value = 50
        while danger_value > 0:
            if time_after_normal_break > 1:
                time_after_normal_break -= 1
                danger_value -= self.danger_range_decrease_by_1s
            elif 0 < time_after_normal_break < 1:
                danger_value -= self.danger_range_decrease_by_1s * time_after_normal_break
                return danger_value

        return 0

    def get_emergency_brake_distance(self, v, acc, vehicle_id):
        if 'car' in vehicle_id:
            distance_react = (v * self.reaktion_time_car) + (0.5 * acc * self.reaktion_time_car ** 2)
            v_after_reaction = v - (acc * self.reaktion_time_car)
            distance_brake = (0.5 * (v_after_reaction ** 2)) / self.emergency_brake_car
        if 'bike' in vehicle_id:
            distance_react = (v * self.reaktion_time_bike) + (0.5 * acc * self.reaktion_time_bike ** 2)
            v_after_reaction = v - (acc * self.reaktion_time_bike)
            distance_brake = (0.5 * (v_after_reaction ** 2)) / self.emergency_brake_bike
        return distance_react + distance_brake

    def get_normal_brake_distance(self, v, acc, vehicle_id):
        if 'car' in vehicle_id:
            distance_react = (v * self.reaktion_time_car) + (0.5 * acc * self.reaktion_time_car ** 2)
            v_after_reaction = v - (acc * self.reaktion_time_car)
            distance_brake = (0.5 * (v_after_reaction ** 2)) / self.normal_brake_car
        if 'bike' in vehicle_id:
            distance_react = (v * self.reaktion_time_bike) + (0.5 * acc * self.reaktion_time_bike ** 2)
            v_after_reaction = v - (acc * self.reaktion_time_bike)
            distance_brake = (0.5 * (v_after_reaction ** 2)) / self.normal_brake_bike
        return distance_react + distance_brake

    def plot_cwa(self, danger_zone, pred_path_own, pred_path_other):
        fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2)
        ax1 = ax[0]
        ax2 = ax[1]

        # plotting bike = own
        ax1.plot(danger_zone.current_pos_1[0], danger_zone.current_pos_1[1], marker="o", markersize=5, color="green")
        ax1.plot(*zip(*pred_path_own), color="green")
        bounding_box_own = Polygon(danger_zone.current_error_bb_1, color="darkgreen", alpha=0.2)
        ax1.add_patch(bounding_box_own)

        # plotting car = other
        ax1.plot(danger_zone.current_pos_2[0], danger_zone.current_pos_2[1], marker="o", markersize=5, color="red")
        ax1.plot(*zip(*pred_path_other), color="red")
        print("danger_zone.current_error_bb_2", danger_zone.current_error_bb_2)
        bounding_box_other = Polygon(danger_zone.current_error_bb_2, color="darkred", alpha=0.2)
        ax1.add_patch(bounding_box_other)

        # plotting danger zone
        for p in danger_zone.zone_vertices:
            if p is not None:
                ax1.plot(p[0], p[1], marker="o", markersize=5, color="black")
        bounding_box_other = Rectangle(danger_zone.zone_vertices[0],
                                       width=danger_zone.zone_vertices[3][0] - danger_zone.zone_vertices[0][0],
                                       height=danger_zone.zone_vertices[1][1] - danger_zone.zone_vertices[0][1],
                                       color="red", alpha=0.1)
        ax1.add_patch(bounding_box_other)

        ax1.set_xlim([480, 530])
        ax1.set_ylim([480, 530])

        # plotting danger-zone matrix
        # ax2.imshow(danger_zone.zone_matrix, cmap='Oranges')
        if danger_zone.zone_matrix_resolution_m < 0.2:
            c = ax2.pcolormesh(danger_zone.zone_matrix, cmap='Oranges')
        else:
            c = ax2.pcolormesh(danger_zone.zone_matrix, edgecolors='k', linewidth=2, cmap='Oranges')
            plt.colorbar(c)
            plt.gca()
        ax2.set_title(str("avg danger value: " + str(danger_zone.danger_value)))

        plt.tight_layout()
        plt.show()
