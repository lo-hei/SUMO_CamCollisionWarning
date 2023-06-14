import math
from copy import deepcopy
from typing import List, Tuple

import traci  # noqa

import simulationClasses.cooperativeAwarenessMessage as Cam
from simulationClasses.GpsModel.simpleGpsModel import SimpleGpsModel
from simulationClasses.TransmissionModel.simpleTransmissionModel import SimpleTransmissionModel


class CamProvider:

    def __init__(self, vehicle):
        self.vehicle = vehicle

        if self.vehicle.get_type() == "car":
            vehicle_factor = 0.4
        elif self.vehicle.get_type() == "bike":
            vehicle_factor = 1

        self.gps_model = SimpleGpsModel("GpsModels-victor-1", vehicle_factor=vehicle_factor)
        self.gps_model.load_model()
        self.transmission_model = SimpleTransmissionModel("TransmissionModel-preTest")
        self.transmission_model.load_model(vehicle_type=self.vehicle.get_type())

        self.update_rate = 1        # in seconds
        self.current_cam = self.generate_cam()
        self.last_cam_generated_time = self.vehicle.simulation_manager.time

    def generate_cam(self) -> Cam.CooperativeAwarenessMessage:
        # generates a new CAM
        cooperative_awareness_message = Cam.CooperativeAwarenessMessage(vehicle=self.vehicle)
        latitude = cooperative_awareness_message.latitude
        longitude = cooperative_awareness_message.longitude

        new_latitude, new_longitude, error = self.gps_model.apply_inaccuracy(latitude=latitude, longitude=longitude)
        self.vehicle.add_cam_path_position(new_latitude, new_longitude)

        cooperative_awareness_message.latitude = new_latitude
        cooperative_awareness_message.longitude = new_longitude
        cooperative_awareness_message.semi_major_confidence = error
        cooperative_awareness_message.semi_minor_confidence = error
        cooperative_awareness_message.semi_major_orientation = 0

        return cooperative_awareness_message

    def get_current_cam(self, position_receiver) -> Tuple[Cam.CooperativeAwarenessMessage, None]:
        # returns the current CAM of the own vehicle

        if position_receiver is None:
            # Then cam_message is requested by yourself
            delivery = True
        else:
            position_self = [self.vehicle.latitude, self.vehicle.longitude]
            distance = math.dist(position_receiver, position_self)
            print("Distance is ", distance)

            if self.transmission_model.apply_uncertainty_delivery(distance=distance):
                delivery = True
            else:
                delivery = False

        if delivery:

            current_time = self.vehicle.simulation_manager.time
            time_cam_available = self.last_cam_generated_time + self.transmission_model.transmission_time

            if (current_time - time_cam_available) > self.update_rate:
                self.current_cam = self.generate_cam()
                self.last_cam_generated_time = current_time
            return self.current_cam

        else:
            return None

