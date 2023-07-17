import math
from copy import deepcopy
from typing import List, Tuple

import traci  # noqa

import simulationClasses.cooperativeAwarenessMessage as Cam
from simulationClasses.GpsModel.simpleGpsModel import SimpleGpsModel
from simulationClasses.TransmissionModel.simpleTransmissionModel import SimpleTransmissionModel


class CamProvider:

    def __init__(self, vehicle, gps_model, transmission_model):
        self.vehicle = vehicle

        # vehicle_factor < 1 means more precision, = 0 means perfect conditions
        if self.vehicle.get_type() == "car":
            vehicle_factor = 1
        elif self.vehicle.get_type() == "bike":
            vehicle_factor = 0.75

        vehicle_factor_transmission_model = 1
        self.use_gps_heading = True
        self.use_gps_speed = True

        self.update_rate = 1        # in seconds

        self.gps_model = SimpleGpsModel(gps_model, vehicle_id=self.vehicle.vehicle_id,
                                        vehicle_factor=vehicle_factor)
        self.gps_model.load_model()
        self.transmission_model = SimpleTransmissionModel(transmission_model,
                                                          vehicle_factor=vehicle_factor_transmission_model)
        self.transmission_model.load_model(vehicle_type=self.vehicle.get_type())

        # self.current_cam = self.generate_cam()
        self.last_cam_generated_time = self.vehicle.simulation_manager.time

    def generate_cam(self) -> Cam.CooperativeAwarenessMessage:
        # generates a new CAM
        cooperative_awareness_message = Cam.CooperativeAwarenessMessage(vehicle=self.vehicle)

        gps_fix = self.gps_model.get_current_fix()
        fix_latitude = gps_fix["latitude"]
        fix_longitude = gps_fix["longitude"]
        fix_time = gps_fix["time"]
        fix_error = gps_fix["error"]
        if self.use_gps_heading:
            if not gps_fix["heading"] is None:
                cooperative_awareness_message.heading = gps_fix["heading"]
        if self.use_gps_speed:
            if not gps_fix["speed"] is None:
                cooperative_awareness_message.speed = gps_fix["speed"]

        self.vehicle.add_cam_path_position(fix_latitude, fix_longitude)

        cooperative_awareness_message.latitude = fix_latitude
        cooperative_awareness_message.longitude = fix_longitude
        cooperative_awareness_message.gps_time = fix_time

        if fix_error is None:
            cooperative_awareness_message.semi_major_confidence = None
            cooperative_awareness_message.semi_minor_confidence = None
            cooperative_awareness_message.semi_major_orientation = None
        else:
            cooperative_awareness_message.semi_major_confidence = fix_error[1]
            cooperative_awareness_message.semi_minor_confidence = fix_error[2]
            cooperative_awareness_message.semi_major_orientation = fix_error[0]

        return cooperative_awareness_message

    def get_current_cam(self, position_receiver) -> Tuple[Cam.CooperativeAwarenessMessage, None]:
        # returns the current CAM of the own vehicle

        current_time = self.vehicle.simulation_manager.time
        time_cam_available = self.last_cam_generated_time + self.transmission_model.transmission_time

        if (current_time - time_cam_available) > self.update_rate:

            if position_receiver is None:
                # Then cam_message is requested by yourself
                delivery = True
            else:
                position_self = [self.vehicle.latitude, self.vehicle.longitude]
                distance = math.dist(position_receiver, position_self)

                if self.transmission_model.apply_uncertainty_delivery(distance=distance):
                    delivery = True
                else:
                    delivery = False

            if delivery:
                self.current_cam = self.generate_cam()
                self.last_cam_generated_time = current_time
                return self.current_cam

            else:
                return None

