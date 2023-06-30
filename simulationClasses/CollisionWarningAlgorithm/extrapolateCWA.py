import math

import numpy as np
from scipy.interpolate import UnivariateSpline

import simulationClasses.CollisionWarningAlgorithm.collisionWarningAlgorithm as cwa
from simulationClasses.utils import helper


class ExtrapolationCWA(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike):
        super(ExtrapolationCWA, self).__init__(bike=bike)

        self.drive_directions = {}  # {vehicle_id: [Points of driveway]]}
        self.extrapolate_frequency = 5  # in Hz
        self.extrapolate_duration = 10  # in sec
        self.look_at_last_n_cams = 1

        self.critical_time_warning = 6
        self.critical_warning = 3

    def interpolate_other_position(self, current_other_cam):
        last_longitude = current_other_cam.longitude
        last_latitude = current_other_cam.latitude
        last_speed = current_other_cam.speed
        last_heading = current_other_cam.heading
        last_longitudinal_acceleration = current_other_cam.longitudinal_acceleration
        last_time = current_other_cam.creation_time

        time_to_interpolate = self.bike.simulation_manager.time - last_time
        interpolated_distanze = (last_speed*time_to_interpolate) + \
                                (0.5 * last_longitudinal_acceleration * time_to_interpolate ** 2)

        if last_heading <= 90 or last_heading >= 270:
            alpha = last_heading
        else:
            alpha = last_heading - 90

        r_long = math.sin(90 - alpha) * interpolated_distanze
        r_lat = math.cos(90 - alpha) * interpolated_distanze

        interpolated_longitude = last_longitude - r_long
        interpolated_latitude = last_latitude - r_lat

        return interpolated_longitude, interpolated_latitude

    def extrapolate_drive_direction(self, last_cams):
        drive_direction = []

        if len(last_cams) > self.look_at_last_n_cams:
            last_cams = last_cams[-self.look_at_last_n_cams:]

        times = [cam.creation_time for cam in last_cams]
        speeds = [cam.speed for cam in last_cams]
        headings = [cam.heading for cam in last_cams]
        accelerations = [cam.longitudinal_acceleration for cam in last_cams]
        longitudes = [cam.longitude for cam in last_cams]
        latitude = [cam.latitude for cam in last_cams]

        extrapolator_speed = UnivariateSpline(times, speeds, k=2)
        speeds_extrapolated = np.array(extrapolator_speed(self.extrapolate_duration))
        speeds_extrapolated[speeds_extrapolated < 0] = 0

        extrapolator_heading = UnivariateSpline(times, speeds, k=2)
        headings_extrapolated = np.array(extrapolator_heading(self.extrapolate_duration))

        extrapolator_acceleration = UnivariateSpline(times, speeds, k=2)
        accelerations_extrapolated = np.array(extrapolator_acceleration(self.extrapolate_duration))
        accelerations_extrapolated[accelerations_extrapolated < 0] = 0

        times_to_extrapolate = list(np.arange(0, self.extrapolate_duration, 1 / self.extrapolate_frequency))

        for time in times_to_extrapolate:

            interpolated_distance = (speeds[-1] * time) + \
                                    (0.5 * accelerations[-1] * time ** 2)

            if headings[-1] <= 90 or headings[-1] >= 270:
                alpha = headings[-1]
            else:
                alpha = headings[-1] - 90

            r_long = math.sin(90 - alpha) * interpolated_distance
            r_lat = math.cos(90 - alpha) * interpolated_distance

            extrapolated_longitude = longitudes[-1] - r_long
            extrapolated_latitude = latitude[-1] - r_lat

            drive_direction.append([extrapolated_latitude, extrapolated_longitude])

        return drive_direction

    def check(self):
        """
        Checking if there is a car in a radius r.
        Using simple interpolation of current position from last CAM with speed, acceleration and heading
        No Acceleration used for assigning risk
        :return:
        """

        for other_vehicle_id, cams in self.bike.received_cams.items():
            if cams[-1]:
                current_other_cam = cams[-1]
            else:
                continue

            own_longitude = self.bike.longitude
            own_latitude = self.bike.latitude

            other_longitude, other_latitude = self.interpolate_other_position(current_other_cam)

            distance = helper.distance(longitude_1=own_longitude, latitude_1=own_latitude,
                                       longitude_2=other_longitude, latitude_2=other_latitude)

            if distance < self.radius_collision:

                self.risk_assessment[other_vehicle_id] = cwa.Risk.Collision
                print("Sending COLLISION to", self.bike.vehicle_id)

            elif distance < self.radius_warning:

                self.risk_assessment[other_vehicle_id] = cwa.Risk.Warning
                print("Sending WARNING to", self.bike.vehicle_id)

            else:
                self.risk_assessment[other_vehicle_id] = cwa.Risk.NoRisk
