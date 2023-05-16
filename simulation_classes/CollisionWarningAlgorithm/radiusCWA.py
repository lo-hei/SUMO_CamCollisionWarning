from simulation_classes import helper
import simulation_classes.CollisionWarningAlgorithm.collisionWarningAlgorithm as cwa
import simulation_classes.CollisionWarningAlgorithm.collisionWarningMessage as cwm


class RadiusCWA(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike, radius):
        super(RadiusCWA, self).__init__(bike=bike)
        self.radius = radius

    def check(self):
        """
        Just checking if there is a car in a radius r.
        No interpolation of current position from last CAM
        No Acceleration used
        :return:
        """

        for other_vehicle_id, cams in self.bike.received_cams.items():
            if cams[-1]:
                current_other_cam = cams[-1]
            else:
                continue

            own_longitude = self.bike.longitude
            own_latitude = self.bike.latitude

            other_longitude = current_other_cam.longitude
            other_latitude = current_other_cam.latitude

            distance = helper.distance(longitude_1=own_longitude, latitude_1=own_latitude,
                                       longitude_2=other_longitude, latitude_2=other_latitude)

            if distance < self.radius:

                collision_warning_message = cwm.CollisionWarningMessage(vehicle_id_1=self.bike.vehicle_id,
                                                                        vehicle_id_2=other_vehicle_id)
                self.collision_warning_messages.append(collision_warning_message)
                self.risk_assessment[other_vehicle_id] = cwa.Risk.SendWarning

            else:
                self.risk_assessment[other_vehicle_id] = cwa.Risk.NoRisk
