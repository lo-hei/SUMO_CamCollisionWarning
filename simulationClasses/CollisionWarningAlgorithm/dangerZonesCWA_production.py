import math
import numpy as np
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


def extrapolate_movement(t_0, lat_0, lon_0, head, times: list, speed: list, acc: list):
    times.insert(0, t_0)
    angle = (head / 180) * math.pi
    d_total = 0

    for i in range(len(times) - 1):
        delta_t = times[i + 1] - times[i]
        d = (speed[i] * delta_t) + (0.5 * acc[i] * (delta_t ** 2))
        d_total += d

    extrap_lat = lat_0 + (d * math.cos(angle))
    extrap_lon = lon_0 + (d * math.sin(angle))
    return extrap_lat, extrap_lon


"""
adapted Version of the DangerZone-CWA
This simpler not-object-orientated-version is not using a seperated Class for the DangerZone
Meant to run on the EVK
"""


class DangerZonesCWA_production(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike):
        super(DangerZonesCWA_production, self).__init__(bike=bike)

        self.plot_after_second = None
        self.last_plot_time = 1

        # {vehicle_id: CAM, v_id: cam, ... }
        self.last_cams_from_vehicles = {}
        # {vehicle_id: (extrap_lat_own, extrap_lon_own, extrap_times_own, extrap_dist_own) ... }
        self.extrapolated_vehicle_paths = {}
        # movement for a given time
        self.movement_own = {"time": [], "speed": [], "acc": [], "position": []}

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
        self.warning_interval = [0.60, 0.8]
        self.collision_interval = [0.8, 1]

        # ---- danger_zone ----

        self.zone_vertices = {}
        self.zone_widths = {}
        self.zone_sizes = {}

    def extrapolate_trajectory(self, lon, lat, v, head, acc, time):
        times_to_extrapolate = np.arange(0, self.extrapolate_time, self.extrapolate_timestep)
        extrapolated_longitude = []
        extrapolated_latitude = []
        extrapolated_times = []
        extrapolated_dist = []

        angle = (head / 180) * math.pi

        for t in times_to_extrapolate:

            if t - times_to_extrapolate[0] < 1:
                # If acc is very low (breaking) the path is not correct predicted because speed getting negative
                # assume bike will only break for maximum 1 seconds, assuming worst case
                d = (v * t) + (0.5 * acc * (t ** 2))
            else:
                slower_speed = v + acc
                if slower_speed < 0:
                    v = 0
                d = (v * 1) + (0.5 * acc * (1 ** 2)) + ((slower_speed) * (t - 1))

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
            if math.dist((self.bike.gps_latitude, self.bike.gps_longitude),
                         (cams[-1].latitude, cams[-1].longitude)) > self.distance_to_attention:
                self.set_risk(vehicle_id_other, no_risk=True)
                print("SKIP, due to too big distance between vehicles")
                continue

            # if heading is wrong between vehicles, skip
            angle_interval_right = ((self.bike.heading - 90 - 0.5 * self.side_angle_to_attention) % 360,
                                    (self.bike.heading - 90 + 0.5 * self.side_angle_to_attention) % 360)
            angle_interval_left = ((self.bike.heading + 90 - 0.5 * self.side_angle_to_attention) % 360,
                                   (self.bike.heading + 90 + 0.5 * self.side_angle_to_attention) % 360)

            in_interval = False

            for angle_interval in [angle_interval_left, angle_interval_right]:
                if angle_interval[0] <= angle_interval[-1]:
                    if angle_interval[0] <= cams[-1].heading <= angle_interval[1]:
                        in_interval = True
                else:
                    if (angle_interval[0] <= cams[-1].heading <= 360) or (0 <= cams[-1].heading <= angle_interval[-1]):
                        in_interval = True

            if not in_interval:
                self.set_risk(vehicle_id_other, no_risk=True)
                print("SKIP, due to wrong heading between vehicles")
                continue

            # add own movement
            self.movement_own["time"].append(self.bike.simulation_manager.time)
            self.movement_own["speed"].append(self.bike.speed)
            self.movement_own["acc"].append(self.bike.longitudinal_acceleration)

            # add extrapolated own position to movement
            t_s_a = zip(self.movement_own["time"], self.movement_own["speed"], self.movement_own["acc"])
            t_s_a = [(t, s, a) for t, s, a in t_s_a if t >= self.bike.gps_time]

            times, speeds, accs = zip(*t_s_a)

            extrap_lat, extrap_lon = extrapolate_movement(t_0=self.bike.gps_time, lon_0=self.bike.gps_longitude,
                                                          lat_0=self.bike.gps_latitude, head=self.bike.heading,
                                                          times=list(times), speed=list(speeds), acc=list(accs))

            current_pos_own = (extrap_lat, extrap_lon)
            self.movement_own["position"].append(current_pos_own)

            # extrapolate trajectory starting in current extrapolated own position
            extrap_lat_own, extrap_lon_own, extrap_times_own, extrap_dist_own = \
                self.extrapolate_trajectory(lon=self.movement_own["position"][-1][1],
                                            lat=self.movement_own["position"][-1][0],
                                            v=self.bike.speed,
                                            head=self.bike.heading,
                                            acc=self.bike.longitudinal_acceleration,
                                            time=self.movement_own["time"][-1])

            # calculate other position and extrapolate trajectory, if new cam
            if new_cam:
                extrap_lat_other, extrap_lon_other, extrap_times_other, extrap_dist_other = \
                    self.extrapolate_trajectory(lon=self.last_cams_from_vehicles[vehicle_id_other].longitude,
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
                                                       list_b=list(zip(extrap_lat_other, extrap_lon_other)))

            # calculate poc (Point of Collision) -- finding intersection on polylines
            poc = None
            for ow in range(len(extrap_lon_own) - 1):
                segment_own = [(extrap_lat_own[ow], extrap_lon_own[ow]),
                               (extrap_lat_own[ow + 1], extrap_lon_own[ow + 1])]

                for ot in range(len(extrap_lon_other) - 1):
                    segment_other = [(extrap_lat_other[ot], extrap_lon_other[ot]),
                                     (extrap_lat_other[ot + 1], extrap_lon_other[ot + 1])]

                    # check if linesegments intersect
                    poc = calculate_segment_intersection(segment_other, segment_own)
                    if poc is not None:
                        break

                if poc is not None:
                    break

            # if trajectories do not intersect at all - no danger for both vehicles
            if poc is None:
                self.set_risk(vehicle_id_other, no_risk=True)
                print("SKIP, due to no intersection of trajectories")
                continue

            # calculate dangerZone_size with poc and the error-ellipses
            error_ellipse_own = self.bike.gps_error
            error_ellipse_other = [self.last_cams_from_vehicles[vehicle_id_other].semi_major_orientation,
                                   self.last_cams_from_vehicles[vehicle_id_other].semi_major_confidence,
                                   self.last_cams_from_vehicles[vehicle_id_other].semi_minor_confidence]

            # if both vehicles have not entered the dangerZone
            outside_dangerzone = False
            if vehicle_id_other in self.zone_widths.keys():
                if (math.dist(current_pos_own, poc) > self.zone_widths[vehicle_id_other][0]) or \
                        (vehicle_id_other not in self.zone_widths.keys()):
                    if math.dist(current_pos_other, poc) > self.zone_widths[vehicle_id_other][1] or \
                            (vehicle_id_other not in self.zone_widths.keys()):
                        outside_dangerzone = True

            if vehicle_id_other not in self.zone_widths.keys() or outside_dangerzone:
                self.create_danger_zone_from_error_ellipses(vehicle_id_other=vehicle_id_other,
                                                            poc=poc, pos_own=current_pos_own,
                                                            pos_other=current_pos_other,
                                                            error_own=error_ellipse_own,
                                                            error_other=error_ellipse_other)

            # calculate time_to_poc for both vehicles
            # for own no if-statement needed, because own is always in front of poc because trajectories intersected and
            # own trajectory starts in current_pos_own

            time_to_poc_own = self.calculate_time_to_poc(
                dist=math.dist(poc, current_pos_own), extrap_dist=extrap_dist_own,
                extrap_times=extrap_times_own
            )

            # check if other is in front of poc or already passed -> if has passed: no Danger

            if math.dist((extrap_lat_other[0], extrap_lon_other[0]), current_pos_other) + \
                    math.dist(current_pos_other, poc) == math.dist((extrap_lat_other[0], extrap_lon_other[0]), poc):
                time_to_poc_other = self.calculate_time_to_poc(
                    dist=math.dist(poc, current_pos_other), extrap_dist=extrap_dist_other,
                    extrap_times=extrap_times_other
                )
            else:
                self.set_risk(vehicle_id_other, no_risk=True)
                continue

            diff_in_ttc = abs(time_to_poc_own - time_to_poc_other)

            # calculate danger_values for both vehicles and add them together
            # if vehicle is in DangerZone -> danger_value = 100

            danger_value_own = self.calculate_danger_value(
                dist=math.dist(poc, current_pos_own),
                dangerZone_width=self.zone_widths[vehicle_id_other][0],
                vehicle_id=self.bike.vehicle_id,
                speed=self.bike.speed,
                acc=self.bike.longitudinal_acceleration,
                extrap_dist=extrap_dist_own, extrap_times=extrap_times_own
            )

            danger_value_other = self.calculate_danger_value(
                dist=math.dist(poc, current_pos_other),
                dangerZone_width=self.zone_widths[vehicle_id_other][1],
                vehicle_id=vehicle_id_other,
                speed=self.last_cams_from_vehicles[vehicle_id_other].speed,
                acc=self.last_cams_from_vehicles[vehicle_id_other].longitudinal_acceleration,
                extrap_dist=extrap_dist_other, extrap_times=extrap_times_other
            )

            # set all CWA-Values
            self.set_danger_zone_area(vehicle_id_other=vehicle_id_other, poc=poc,
                                      pos_own=current_pos_own, pos_other=current_pos_other, error_own=error_ellipse_own)

            danger_value = danger_value_own + danger_value_other
            self.set_risk(vehicle_id=vehicle_id_other, diff_in_ttc=diff_in_ttc, danger_value=danger_value)

    def set_risk(self, vehicle_id, diff_in_ttc=None, danger_value=0, no_risk=False):

        if no_risk:
            self.risk_assessment[vehicle_id] = cwa.Risk.NoRisk
        else:
            prob_collision = (1 / (1 + diff_in_ttc)) * (danger_value / 200)

            if self.warning_interval[0] <= prob_collision <= self.warning_interval[1]:
                self.risk_assessment[vehicle_id] = cwa.Risk.Warning
            elif self.collision_interval[0] <= prob_collision <= self.collision_interval[1]:
                self.risk_assessment[vehicle_id] = cwa.Risk.Collision
            else:
                self.risk_assessment[vehicle_id] = cwa.Risk.NoRisk

    def calculate_time_to_poc(self, dist, extrap_dist, extrap_times):
        time_to_poc = get_interpolated_value(dist, extrap_dist, extrap_times)
        return time_to_poc

    def create_danger_zone_from_error_ellipses(self, vehicle_id_other, poc, pos_own, pos_other, error_own, error_other):
        c = math.acos(math.dist((poc[0], pos_own[1]), poc) / math.dist(poc, pos_own)) / math.pi * 180
        d = math.acos(math.dist((poc[0], pos_other[1]), poc) / math.dist(poc, pos_other)) / math.pi * 180

        a_1 = abs(error_own[0] - c) % 90
        b_1 = abs(error_own[0] - d) % 90

        a_2 = abs(error_other[0] - c) % 90
        b_2 = abs(error_other[0] - d) % 90

        width_a_1 = math.dist(pos_own,
                              calculate_ellipse_point(center=pos_own, theta=a_1,
                                                      error_ellipse=error_own))

        width_b_1 = math.dist(pos_own,
                              calculate_ellipse_point(center=pos_own, theta=b_1,
                                                      error_ellipse=error_own))

        width_a_2 = math.dist(pos_other,
                              calculate_ellipse_point(center=pos_other, theta=a_2,
                                                      error_ellipse=error_other))
        width_b_2 = math.dist(pos_other,
                              calculate_ellipse_point(center=pos_other, theta=b_2,
                                                      error_ellipse=error_other))

        self.zone_widths[vehicle_id_other] = [width_a_1 + width_b_2, width_b_1 + width_a_2]

        # calculate the four vertices
        vector_trajectory_1 = (poc[0] - pos_own[0], poc[1] - pos_own[1])
        unit_vector_trajectory_1 = np.array(vector_trajectory_1) / np.linalg.norm(vector_trajectory_1)

        vector_trajectory_2 = (poc[0] - pos_other[0], poc[1] - pos_other[1])
        unit_vector_trajectory_2 = np.array(vector_trajectory_2) / np.linalg.norm(vector_trajectory_2)

        point_1 = np.array(poc) + (-1) * unit_vector_trajectory_1 * self.zone_widths[vehicle_id_other][0] + \
                  (-1) * unit_vector_trajectory_2 * self.zone_widths[vehicle_id_other][1]

        point_2 = np.array(poc) + (-1) * unit_vector_trajectory_1 * self.zone_widths[vehicle_id_other][0] + \
                  unit_vector_trajectory_2 * self.zone_widths[vehicle_id_other][1]

        point_3 = np.array(poc) + unit_vector_trajectory_1 * self.zone_widths[vehicle_id_other][0] + \
                  unit_vector_trajectory_2 * self.zone_widths[vehicle_id_other][1]

        point_4 = np.array(poc) + unit_vector_trajectory_1 * self.zone_widths[vehicle_id_other][0] + \
                  (-1) * unit_vector_trajectory_2 * self.zone_widths[vehicle_id_other][1]

        self.zone_vertices[vehicle_id_other] = [point_1, point_2, point_3, point_4]

    def set_danger_zone_area(self, vehicle_id_other, poc, pos_own, pos_other, error_own):
        c = math.acos(math.dist((poc[0], pos_own[1]), poc) / math.dist(poc, pos_own)) / math.pi * 180
        d = math.acos(math.dist((poc[0], pos_other[1]), poc) / math.dist(poc, pos_other)) / math.pi * 180

        a_1 = abs(error_own[0] - c) % 90
        b_1 = abs(error_own[0] - d) % 90

        h = (math.sin((180 - b_1 - a_1) / 180 * math.pi) * 2 * self.zone_widths[vehicle_id_other][1])
        self.zone_sizes[vehicle_id_other] = abs(h * 2 * self.zone_widths[vehicle_id_other][0])

    def calculate_danger_value(self, dist, dangerZone_width, vehicle_id, speed, acc, extrap_dist, extrap_times):
        if dist > extrap_dist[-1]:
            return 0

        # if position is in DangerZone. return 100
        if dist <= dangerZone_width:
            return 100

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