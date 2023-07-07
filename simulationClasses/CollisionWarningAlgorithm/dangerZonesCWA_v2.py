import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Ellipse
from scipy.stats import norm

from simulationClasses.utils import helper
import simulationClasses.CollisionWarningAlgorithm.collisionWarningAlgorithm as cwa


def calculate_ellipse_point(center, theta, error_ellipse):
    theta = (theta / 180) * math.pi

    dividend = (error_ellipse[1] * error_ellipse[2])
    divisor = (math.sqrt(error_ellipse[2] ** 2 + (error_ellipse[1] ** 2 * math.tan(theta) ** 2)))
    x_ellipse = dividend / divisor

    if not -math.pi / 2 < theta < math.pi / 2:
        x_ellipse = x_ellipse * (-1)

    y_ellipse = x_ellipse * math.tan(theta)
    return center[0] + x_ellipse, center[1] + y_ellipse


def get_interpolated_value(value_a, list_a, list_b):
    # returns the interpolated value_b from list_b
    if not list_a[0] <= value_a <= list_a[-1]:
        return None

    higher_than_a = len([d for d in list_a if d >= value_a])

    lower_b = list_b[len(list_a) - 1 - higher_than_a]
    higher_b = list_b[len(list_a) - 1 - higher_than_a + 1]

    lower_a = list_a[len(list_a) - 1 - higher_than_a]
    higher_a = list_a[len(list_a) - 1 - higher_than_a + 1]

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

    def __init__(self, p_1, p_2, poc, error_ellipse_1, error_ellipse_2):
        # 1 = own, 2 = other
        self.current_pos_1 = p_1
        self.current_pos_2 = p_2

        self.current_error_1 = error_ellipse_1
        self.current_error_2 = error_ellipse_2
        self.poc = poc

        self.time_to_poc_1 = []
        self.time_to_poc_2 = []

        self.zone_vertices = []  # [edge_1, edge_2, edge_3, edge_4]
        self.zone_width = []  # [distance_to_own, distance_to_other]
        self.danger_zone_size = []
        self.create_danger_zone_from_error_ellipses()

        self.danger_value = []
        self.danger_value_1 = []
        self.danger_value_2 = []

        self.update_times = []
        self.diff_in_ttc = []
        self.prob_collision = []

    def update(self, p_1, p_2, poc, error_ellipse_1, error_ellipse_2):
        self.current_pos_1 = p_1
        self.current_pos_2 = p_2
        self.current_error_1 = error_ellipse_1
        self.current_error_2 = error_ellipse_2
        self.poc = poc

        if math.dist(p_1, poc) > self.zone_width[-1][0]:
            if math.dist(p_2, poc) > self.zone_width[-1][1]:
                self.create_danger_zone_from_error_ellipses()

    def create_danger_zone_from_error_ellipses(self):
        c = math.acos(math.dist((self.poc[0], self.current_pos_1[1]), self.poc) /
                      math.dist(self.poc, self.current_pos_1)) / math.pi * 180
        d = math.acos(math.dist((self.poc[0], self.current_pos_2[1]), self.poc) /
                      math.dist(self.poc, self.current_pos_2)) / math.pi * 180

        a_1 = abs(self.current_error_1[0] - c) % 90
        b_1 = abs(self.current_error_1[0] - d) % 90

        a_2 = abs(self.current_error_2[0] - c) % 90
        b_2 = abs(self.current_error_2[0] - d) % 90

        width_a_1 = math.dist(self.current_pos_1,
                              calculate_ellipse_point(center=self.current_pos_1, theta=a_1,
                                                      error_ellipse=self.current_error_1))

        width_b_1 = math.dist(self.current_pos_1,
                              calculate_ellipse_point(center=self.current_pos_1, theta=b_1,
                                                      error_ellipse=self.current_error_1))

        width_a_2 = math.dist(self.current_pos_2,
                              calculate_ellipse_point(center=self.current_pos_2, theta=a_2,
                                                      error_ellipse=self.current_error_2))
        width_b_2 = math.dist(self.current_pos_2,
                              calculate_ellipse_point(center=self.current_pos_2, theta=b_2,
                                                      error_ellipse=self.current_error_2))

        self.zone_width.append([width_a_1 + width_b_2, width_b_1 + width_a_2])

        # calculate the four vertices
        vector_trajectory_1 = (self.poc[0] - self.current_pos_1[0], self.poc[1] - self.current_pos_1[1])
        unit_vector_trajectory_1 = np.array(vector_trajectory_1) / np.linalg.norm(vector_trajectory_1)

        vector_trajectory_2 = (self.poc[0] - self.current_pos_2[0], self.poc[1] - self.current_pos_2[1])
        unit_vector_trajectory_2 = np.array(vector_trajectory_2) / np.linalg.norm(vector_trajectory_2)

        point_1 = np.array(self.poc) + (-1) * unit_vector_trajectory_1 * self.zone_width[-1][0] + \
                  (-1) * unit_vector_trajectory_2 * self.zone_width[-1][1]

        point_2 = np.array(self.poc) + (-1) * unit_vector_trajectory_1 * self.zone_width[-1][0] + \
                  unit_vector_trajectory_2 * self.zone_width[-1][1]

        point_3 = np.array(self.poc) + unit_vector_trajectory_1 * self.zone_width[-1][0] + \
                  unit_vector_trajectory_2 * self.zone_width[-1][1]

        point_4 = np.array(self.poc) + unit_vector_trajectory_1 * self.zone_width[-1][0] + \
                  (-1) * unit_vector_trajectory_2 * self.zone_width[-1][1]

        self.zone_vertices.append([point_1, point_2, point_3, point_4])
        h = (math.sin((180 - b_1 - a_1) / 180 * math.pi) * 2 * self.zone_width[-1][1])

        self.danger_zone_size.append(abs(h * 2 * self.zone_width[-1][0]))

    def get_time_to_poc_1(self):
        return self.time_to_poc_1[-1]

    def get_time_to_poc_2(self):
        return self.time_to_poc_2[-1]

    def set_time_to_poc_1(self, time_to_poc_1):
        self.time_to_poc_1.append(time_to_poc_1)

    def set_time_to_poc_2(self, time_to_poc_2):
        self.time_to_poc_2.append(time_to_poc_2)

    def get_danger_zone_size(self):
        return self.danger_zone_size[-1]

    def get_zone_vertices(self):
        return self.zone_vertices[-1]

    def get_danger_zone_size(self):
        return self.danger_zone_size[-1]

    def get_danger_value(self):
        return self.danger_value[-1]

    def set_danger_value_1(self, dv_1):
        self.danger_value_1.append(dv_1)

    def set_danger_value_2(self, dv_2):
        self.danger_value_2.append(dv_2)

    def set_danger_value(self):
        self.danger_value.append(self.danger_value_1[-1] + self.danger_value_2[-1])

    def get_zone_vertices(self):
        return self.zone_vertices[-1]


class DangerZonesCWA_v2(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike):
        super(DangerZonesCWA_v2, self).__init__(bike=bike)

        self.plot_after_second = 2
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

        self.danger_range_decrease_by_1s = 5
        self.warning_interval = [15, 20]
        self.collision_interval = [20, 100]

        self.danger_zones = {}  # {vehicle_id: DangerZone, v_id: dZ, ... }

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
            current_pos_own = get_interpolated_value(value_a=self.bike.simulation_manager.time,
                                                     list_a=extrap_times_own,
                                                     list_b=list(zip(extrap_lon_own, extrap_lat_own)))

            # calculate other position and extrapolate trajectory, if new cam
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

            current_pos_other = get_interpolated_value(value_a=self.bike.simulation_manager.time,
                                                       list_a=extrap_times_other,
                                                       list_b=list(zip(extrap_lon_other, extrap_lat_other)))

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

            # calculate dangerZone_size with poc and the error-ellipses
            error_ellipse_own = self.bike.gps_error
            error_ellipse_other = [self.last_cams_from_vehicles[vehicle_id_other].semi_major_orientation,
                                   self.last_cams_from_vehicles[vehicle_id_other].semi_major_confidence,
                                   self.last_cams_from_vehicles[vehicle_id_other].semi_minor_confidence]

            if not vehicle_id_other in self.danger_zones.keys():
                danger_zone = DangerZone(p_1=current_pos_own, p_2=current_pos_other,
                                         poc=poc, error_ellipse_1=error_ellipse_own,
                                         error_ellipse_2=error_ellipse_other)
                self.danger_zones[vehicle_id_other] = danger_zone
            else:
                self.danger_zones[vehicle_id_other].update(p_1=current_pos_own, p_2=current_pos_other,
                                                           poc=poc, error_ellipse_1=error_ellipse_own,
                                                           error_ellipse_2=error_ellipse_other)

            # calculate time_to_poc for both vehicles
            self.danger_zones[vehicle_id_other].set_time_to_poc_1(
                self.calculate_time_to_poc(dist=math.dist(poc, current_pos_own),
                                           extrap_dist=extrap_dist_other, extrap_times=extrap_times_other))
            self.danger_zones[vehicle_id_other].set_time_to_poc_2(
                self.calculate_time_to_poc(dist=math.dist(poc, current_pos_other),
                                           extrap_dist=extrap_dist_other, extrap_times=extrap_times_other))

            # calculate danger_values for both vehicles and add them together
            self.danger_zones[vehicle_id_other].set_danger_value_1(
                self.calculate_danger_value(dist=math.dist(poc, current_pos_own),
                                            vehicle_id=self.bike.vehicle_id, speed=self.bike.speed,
                                            acc=self.bike.longitudinal_acceleration,
                                            extrap_dist=extrap_dist_own, extrap_times=extrap_times_own))

            self.danger_zones[vehicle_id_other].set_danger_value_2(
                self.calculate_danger_value(dist=math.dist(poc, current_pos_other), vehicle_id=vehicle_id_other,
                                            speed=self.last_cams_from_vehicles[vehicle_id_other].speed,
                                            acc=self.last_cams_from_vehicles[
                                                vehicle_id_other].longitudinal_acceleration,
                                            extrap_dist=extrap_dist_other, extrap_times=extrap_times_other))

            self.danger_zones[vehicle_id_other].set_danger_value()

            # if vehicle has passed the poc in interpolation, no time_to_poc can be calculated, but danger is still
            # possible because no new cam received
            if (self.danger_zones[vehicle_id_other].get_time_to_poc_1() is None) or \
                    (self.danger_zones[vehicle_id_other].get_time_to_poc_2() is None):
                diff_in_ttc = 0
            else:
                diff_in_ttc = abs(self.danger_zones[vehicle_id_other].get_time_to_poc_1() -
                                  self.danger_zones[vehicle_id_other].get_time_to_poc_2())
            self.danger_zones[vehicle_id_other].diff_in_ttc.append(diff_in_ttc)

            self.danger_zones[vehicle_id_other].update_times.append(self.bike.simulation_manager.time)

            self.danger_zones[vehicle_id_other].prob_collision.append(
                (1 / (1 + diff_in_ttc)) \
                * (1 / math.sqrt(self.danger_zones[vehicle_id_other].get_danger_zone_size())) \
                * self.danger_zones[vehicle_id_other].get_danger_value())

            if self.warning_interval[0] <= self.danger_zones[vehicle_id_other].prob_collision[-1] <= self.warning_interval[1]:
                self.risk_assessment[vehicle_id_other] = cwa.Risk.Warning
            elif self.collision_interval[0] <= self.danger_zones[vehicle_id_other].prob_collision[-1] <= self.collision_interval[1]:
                self.risk_assessment[vehicle_id_other] = cwa.Risk.Collision
            self.risk_assessment_history.append([self.bike.simulation_manager.time, self.risk_assessment])

            if self.plot_after_second:
                if (self.bike.simulation_manager.time - self.last_plot_time) > self.plot_after_second:
                    print("Plotting CWA for time", self.bike.simulation_manager.time)
                    self.plot_cwa(self.danger_zones[vehicle_id_other],
                                  zip(extrap_lon_own, extrap_lat_own),
                                  zip(extrap_lon_other, extrap_lat_other))
                    self.last_plot_time = self.bike.simulation_manager.time

    def calculate_time_to_poc(self, dist, extrap_dist, extrap_times):
        time_to_poc = get_interpolated_value(dist, extrap_dist, extrap_times)
        return time_to_poc

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
        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
        ax1 = ax

        # plotting bike = own
        ax1.plot(danger_zone.current_pos_1[0], danger_zone.current_pos_1[1], marker="o", markersize=5, color="green")
        ax1.plot(*zip(*pred_path_own), color="green")
        error_ellipse_1 = Ellipse(xy=danger_zone.current_pos_1, width=2 * danger_zone.current_error_1[1],
                                  height=2 * danger_zone.current_error_1[2], angle=danger_zone.current_error_1[0],
                                  color="darkgreen", alpha=0.2)
        ax1.add_patch(error_ellipse_1)

        # plotting car = other
        ax1.plot(danger_zone.current_pos_2[0], danger_zone.current_pos_2[1], marker="o", markersize=5, color="red")
        ax1.plot(*zip(*pred_path_other), color="red")
        error_ellipse_2 = Ellipse(xy=danger_zone.current_pos_2, width=2 * danger_zone.current_error_2[1],
                                  height=2 * danger_zone.current_error_2[2], angle=danger_zone.current_error_2[0],
                                  color="darkred", alpha=0.2)
        ax1.add_patch(error_ellipse_2)

        # plotting danger zone
        for p in danger_zone.get_zone_vertices():
            if p is not None:
                ax1.plot(p[0], p[1], marker="o", markersize=5, color="black")
        bounding_box_zone = Polygon(danger_zone.get_zone_vertices(), color="darkred", alpha=0.2)
        ax1.add_patch(bounding_box_zone)

        ax1.set_xlim([460, 520])
        ax1.set_ylim([490, 550])

        plt.tight_layout()
        plt.show()
