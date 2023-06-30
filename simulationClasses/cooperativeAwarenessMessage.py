import traci  # noqa
import time


class CooperativeAwarenessMessage:

    def __init__(self, vehicle):
        self.creation_time = vehicle.simulation_manager.time
        self.generation_delta_time = 0
        self.gps_time = None
        self.vehicle_type = vehicle.get_type()

        self.cam_id = vehicle.vehicle_id + "_" + str(self.creation_time)
        self.vehicle_id = vehicle.vehicle_id

        # Basic Container
        self.latitude = vehicle.latitude
        self.longitude = vehicle.longitude
        self.semi_major_orientation = None
        self.semi_major_confidence = None
        self.semi_minor_confidence = None
        self.altitude = 0

        # High frequency container
        self.heading = vehicle.heading
        self.speed = vehicle.speed
        self.drive_direction = 1
        self.longitudinal_acceleration = vehicle.longitudinal_acceleration
        self.vehicle_length = None
        self.vehicle_width = None
        self.curvature = 0
        self.curvature_confidence = 0
        self.yaw_rate = 0
