import simulationClasses.Vehicles.vehicleClass as vehicle


class Car(vehicle.Vehicle):

    def __init__(self, vehicle_id, simulation_manager, gps_model, transmission_model):
        super(Car, self).__init__(vehicle_id=vehicle_id, simulation_manager=simulation_manager,
                                  gps_model=gps_model, transmission_model=transmission_model)
        self.type = "Car"
        self.send_cams = []

    def update(self):
        # everything that has to happen in one simulation-step for a car

        # updating real attributes of vehicle
        self.update_vehicle_attributes()

        # updating own cam message in cam_provider
        self.cam_provider.get_current_cam(position_receiver=None)

    def get_type(self):
        return "car"

