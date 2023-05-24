from typing import Dict

import traci  # noqa

import simulationClasses.camProvider as camProvider


class Vehicle:

    def __init__(self, vehicle_id: str, simulation_manager):
        self.vehicle_id = vehicle_id
        self.simulation_manager = simulation_manager

        self.longitude = None
        self.latitude = None
        self.heading = None
        self.speed = None
        self.longitudinal_acceleration = None
        self.vehicle_length = None
        self.vehicle_width = None
        self.update_vehicle_attributes()

        self.cam_provider = camProvider.CamProvider(self)

    def update_vehicle_attributes(self):
        self.longitude = traci.vehicle.getPosition(self.vehicle_id)[0]
        self.latitude = traci.vehicle.getPosition(self.vehicle_id)[1]
        self.heading = traci.vehicle.getAngle(self.vehicle_id)
        self.speed = traci.vehicle.getSpeed(self.vehicle_id)
        self.longitudinal_acceleration = traci.vehicle.getAcceleration(self.vehicle_id)
        self.vehicle_length = traci.vehicle.getLength(self.vehicle_id)
        self.vehicle_width = traci.vehicle.getWidth(self.vehicle_id)
