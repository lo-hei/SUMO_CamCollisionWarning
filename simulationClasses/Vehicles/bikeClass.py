from typing import List

import simulationClasses.Vehicles.vehicleClass as v
import simulationClasses.CollisionWarningAlgorithm.radiusCWA as radiusCWA
import simulationClasses.CollisionWarningAlgorithm.radiusInterpolationCWA as radiusInterpolationCWA
import simulationClasses.cooperativeAwarenessMessage as cam


class Bike(v.Vehicle):

    def __init__(self, vehicle_id, simulation_manager, gps_model, transmission_model):
        super(Bike, self).__init__(vehicle_id=vehicle_id, simulation_manager=simulation_manager,
                                   gps_model=gps_model, transmission_model=transmission_model)
        self.type = "Bike"
        self.cwa = None

        self.send_cams = []
        self.received_cams = {}  # {vehicle_id: [cam_0, cam_1 ...]}

    def update(self):
        # everything that has to happen in one simulation-step

        # updating real attributes of vehicle
        self.update_vehicle_attributes()

        # updating own cam message in cam_provider
        self.update_send_cam()

        # asking for new CAMs in cam_provider
        self.update_received_cams()

        # updating the CWA
        self.cwa.check()

    def get_type(self):
        return "bike"

    def update_received_cams(self):
        # asking every other Vehicle.cam_provider for new CAMs
        for vehicle_id, vehicle in self.simulation_manager.activeVehicles.items():

            if vehicle_id == self.vehicle_id:
                continue

            position_self = [self.latitude, self.longitude]
            current_vehicle_cam = vehicle.cam_provider.get_current_cam(position_receiver=position_self)
            if current_vehicle_cam:

                if vehicle_id not in self.received_cams.keys():
                    self.received_cams[vehicle_id] = [current_vehicle_cam]

                # TODO: check if CAM is newer
                elif not self.received_cams[vehicle_id][-1].cam_id == current_vehicle_cam.cam_id:
                    self.received_cams[vehicle_id].append(current_vehicle_cam)

    def update_send_cam(self):
        # updates the own cams of the vehicle
        current_cam = self.cam_provider.get_current_cam(position_receiver=None)
        if len(self.send_cams) > 0:
            if not self.send_cams[-1].cam_id == current_cam.cam_id:
                self.send_cams.append(current_cam)

    def get_current_received_cams(self) -> List[cam.CooperativeAwarenessMessage]:
        # returns the current CAM from every other Vehicle
        current_cams = []
        for vehicle_id, cam_list in self.received_cams.items():
            if cam_list[-1]:
                current_cams.append(cam_list[-1])
        return current_cams

    def get_all_received_vehicle_cams(self, vehicle_id) -> List[cam.CooperativeAwarenessMessage]:
        # returns all CAMs received from one vehicle
        return self.received_cams[vehicle_id]
