import random
import time

import traci  # noqa

from simulation_classes import helper
from simulation_classes.Vehicles.bikeClass import Bike
from simulation_classes.Vehicles.carClass import Car
from simulation_classes.CollisionWarningAlgorithm.collisionWarningAlgorithm import Risk


class SimulationManager:

    def __init__(self, step_length):
        self.activeVehicles = {}        # {vehicle_id: Vehicle}
        self.step_length = step_length
        self.time = 0
        self.dangerous_situation = False

        self.badVehicles = []
        self.goodVehicles = []

    def create_bad_vehicles(self, rate=0.8):

        # update badVehicles and goodVehicles and remove inactive ones
        self.badVehicles = [v for v in self.badVehicles if v in self.activeVehicles]
        self.goodVehicles = [v for v in self.goodVehicles if v in self.activeVehicles]

        # get new spawned vehicles
        newVehicles = [v for v in self.activeVehicles if v not in self.badVehicles and v not in self.goodVehicles]

        if newVehicles:
            for v in newVehicles:
                if random.uniform(0, 1) < rate:
                    self.badVehicles.append(v)
                    traci.vehicle.setSpeedMode(v, 32)

                    # setParameter(self, objID, param, value)

                    # This value causes vehicles and pedestrians to ignore foe vehicles that have right-of-way with the
                    # probability. The check is performed anew every simulation step. (range [0,1]).
                    traci.vehicle.setParameter(v, "jmIgnoreFoeProb", 0.1)

                    # This value causes vehicles to ignore foe vehicles and pedestrians that have already entered a
                    # junction with the given probability. The check is performed anew every simulation step.
                    traci.vehicle.setParameter(v, "jmIgnoreJunctionFoeProb", 0.1)

                    # Willingness of drivers to impede vehicles with higher priority.
                    traci.vehicle.setParameter(v, "impatience", 0.5)

                    helper.color_vehicle_red(v)
                else:
                    self.goodVehicles.append(v)

    def is_dangerous_situation(self):
        return self.dangerous_situation

    def looking_for_new_vehicles(self):
        activeVehicles_ids = traci.vehicle.getIDList()

        for v_id in activeVehicles_ids:
            if v_id not in self.activeVehicles:
                # create new Vehicle
                if "bike" in v_id:
                    # create new bike
                    new_bike = Bike(vehicle_id=v_id, simulation_manager=self)
                    self.activeVehicles[v_id] = new_bike
                    helper.color_vehicle_green(v_id)
                elif "car" in v_id:
                    # create new car
                    new_car = Car(vehicle_id=v_id, simulation_manager=self)
                    self.activeVehicles[v_id] = new_car
                    helper.color_vehicle_blue(v_id)
                else:
                    print("Vehicle-Type not found: ", v_id)

    def deleting_inactive_vehicles(self):
        activeVehicles_ids = traci.vehicle.getIDList()
        current_ids = self.activeVehicles.keys()
        inactive_ids = [i for i in current_ids if i not in activeVehicles_ids]
        for inactive in inactive_ids:
            del self.activeVehicles[inactive]

    def update_active_vehicles(self):
        for vehicle_id in self.activeVehicles.keys():
            self.activeVehicles[vehicle_id].update()
            if "bike" in vehicle_id:
                if self.activeVehicles[vehicle_id].cwa.get_current_risk().value >= Risk.Collision.value:
                    self.dangerous_situation = True
                    helper.color_vehicle_red(vehicle_id)
                elif self.activeVehicles[vehicle_id].cwa.get_current_risk().value >= Risk.SendWarning.value:
                    self.dangerous_situation = True
                    helper.color_vehicle_yellow(vehicle_id)
                else:
                    helper.color_vehicle_green(vehicle_id)



    def simulationStep(self):
        # update
        self.looking_for_new_vehicles()
        self.deleting_inactive_vehicles()
        self.update_active_vehicles()

        '''
        if self.dangerous_situation:
            time.sleep(0.2)
        '''

        # self.create_bad_vehicles(rate=0.9)
        self.time += self.step_length

