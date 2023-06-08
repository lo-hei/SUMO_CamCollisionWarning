import math

from simulationClasses.utils import helper
import simulationClasses.CollisionWarningAlgorithm.collisionWarningAlgorithm as cwa
import simulationClasses.CollisionWarningAlgorithm.collisionWarningMessage as cwm


class RadiusInterpolateCWA(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike, radius):
        super(RadiusInterpolateCWA, self).__init__(bike=bike)
        self.radius = radius

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

            if distance < self.radius:

                collision_warning_message = cwm.CollisionWarningMessage(vehicle_id_1=self.bike.vehicle_id,
                                                                        vehicle_id_2=other_vehicle_id)
                self.collision_warning_messages.append(collision_warning_message)
                self.risk_assessment[other_vehicle_id] = cwa.Risk.SendWarning

                print("Sending WARNING to", self.bike.vehicle_id)

            else:
                self.risk_assessment[other_vehicle_id] = cwa.Risk.NoRisk
