import time
from typing import Dict

import traci  # noqa

import simulationClasses.camProvider as camProvider


class Vehicle:

    def __init__(self, vehicle_id: str, simulation_manager):
        self.vehicle_id = vehicle_id
        self.simulation_manager = simulation_manager

        self.latitude = None
        self.longitude = None

        self.gps_latitude = None
        self.gps_longitude = None
        self.gps_time = None
        self.gps_error = None

        self.heading = None
        self.speed = None
        self.longitudinal_acceleration = None
        self.vehicle_length = None
        self.vehicle_width = None

        self.real_path = []
        self.gps_path = []
        self.cam_path = []

        self.cam_provider = camProvider.CamProvider(self)
        # self.update_vehicle_attributes()

    def update_vehicle_attributes(self):
        self.latitude = traci.vehicle.getPosition(self.vehicle_id)[1]
        self.longitude = traci.vehicle.getPosition(self.vehicle_id)[0]

        is_new_fix = self.cam_provider.gps_model.update_current_fix(self.latitude, self.longitude, self.simulation_manager.time)

        if is_new_fix:
            current_fix = self.cam_provider.gps_model.get_current_fix()
            self.gps_latitude = current_fix["latitude"]
            self.gps_longitude = current_fix["longitude"]
            self.gps_time = current_fix["time"]
            self.gps_error = current_fix["error"]
            self.gps_path.append([current_fix["latitude"], current_fix["longitude"], current_fix["time"]])

        self.heading = traci.vehicle.getAngle(self.vehicle_id)
        self.speed = traci.vehicle.getSpeed(self.vehicle_id)
        self.longitudinal_acceleration = traci.vehicle.getAcceleration(self.vehicle_id)
        self.vehicle_length = traci.vehicle.getLength(self.vehicle_id)
        self.vehicle_width = traci.vehicle.getWidth(self.vehicle_id)

        self.real_path.append([self.latitude, self.longitude, self.simulation_manager.time])

    def add_cam_path_position(self, latitude, longitude):
        self.cam_path.append([latitude, longitude, self.simulation_manager.time])
