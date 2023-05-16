from copy import deepcopy
from typing import List, Tuple

import traci  # noqa

import simulation_classes.cooperativeAwarenessMessage as cam
# from simulation_classes.GpsModel.simpleGpsModel import SimpleGpsModel
# from simulation_classes.TransmissionModel.simpleTranmissionModel import SimpleTranmissionModel


class CamProvider:

    def __init__(self, vehicle):
        self.vehicle = vehicle

        self.update_rate = 1        # in seconds
        self.current_cam = self.generate_cam()
        self.last_cam_generated_time = self.vehicle.simulation_manager.time

        # self.gps_model = SimpleGpsModel()
        # self.transmission_model = SimpleTranmissionModel()

    def generate_cam(self) -> cam.CooperativeAwarenessMessage:
        # generates a new CAM
        cooperative_awareness_message = cam.CooperativeAwarenessMessage(vehicle=self.vehicle)
        return cooperative_awareness_message

    def get_current_cam(self) -> Tuple[cam.CooperativeAwarenessMessage, None]:
        # returns the current CAM of the own vehicle

        # here transmission-Model
        in_transmission_range = True

        if in_transmission_range:

            current_time = self.vehicle.simulation_manager.time
            if (current_time - self.last_cam_generated_time) > self.update_rate:
                self.current_cam = self.generate_cam()
                self.last_cam_generated_time = current_time
            return self.current_cam

        else:
            return None

