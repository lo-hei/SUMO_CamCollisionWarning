from copy import deepcopy
from typing import List, Tuple

import traci  # noqa

import simulationClasses.cooperativeAwarenessMessage as Cam
from simulationClasses.GpsModel.simpleGpsModel import SimpleGpsModel


class CamProvider:

    def __init__(self, vehicle):
        self.vehicle = vehicle

        if self.vehicle.get_type() == "car":
            vehicle_factor = 0.1
        elif self.vehicle.get_type() == "bike":
            vehicle_factor = 1

        self.gps_model = SimpleGpsModel("GpsModels-internal", vehicle_factor=vehicle_factor)
        self.gps_model.load_model()
        # self.transmission_model = SimpleTransmissionModel("TransmissionModels-internal")
        # self.transmission_model.load_model()

        self.update_rate = 1        # in seconds
        self.current_cam = self.generate_cam()
        self.last_cam_generated_time = self.vehicle.simulation_manager.time

    def generate_cam(self) -> Cam.CooperativeAwarenessMessage:
        # generates a new CAM
        cooperative_awareness_message = Cam.CooperativeAwarenessMessage(vehicle=self.vehicle)
        latitude = cooperative_awareness_message.latitude
        longitude = cooperative_awareness_message.longitude

        new_latitude, new_longitude = self.gps_model.apply_inaccuracy(latitude=latitude, longitude=longitude)

        cooperative_awareness_message.latitude = new_latitude
        cooperative_awareness_message.longitude = new_longitude
        return cooperative_awareness_message

    def get_current_cam(self) -> Tuple[Cam.CooperativeAwarenessMessage, None]:
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

