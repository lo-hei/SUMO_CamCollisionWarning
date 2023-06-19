from simulationClasses.utils import helper
import simulationClasses.CollisionWarningAlgorithm.collisionWarningAlgorithm as cwa


class RadiusCWA(cwa.CollisionWarningAlgorithm):

    def __init__(self, bike):
        super(RadiusCWA, self).__init__(bike=bike)

        self.radius_warning = 6
        self.radius_collision = 3

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

            if distance < self.radius_collision:

                self.risk_assessment[other_vehicle_id] = cwa.Risk.Collision
                print("Sending COLLISION to", self.bike.vehicle_id)

            elif distance < self.radius_warning:

                self.risk_assessment[other_vehicle_id] = cwa.Risk.Warning
                print("Sending WARNING to", self.bike.vehicle_id)

            else:
                self.risk_assessment[other_vehicle_id] = cwa.Risk.NoRisk
