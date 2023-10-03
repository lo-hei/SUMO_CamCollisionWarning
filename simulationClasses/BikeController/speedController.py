import keyboard

import traci  # noqa

from simulationClasses.BikeController.controller import Controller


def check_for_keyboard_input():
    if keyboard.is_pressed("w"):
        print("SPEED UP !!!")
        return "speed_up"
    elif keyboard.is_pressed("s"):
        print("SLOW DOWN !!!")
        return "slow_down"
    else:
        return False


"""
Class to control speed of a vehicle during runtime
"""


class SpeedController(Controller):

    def __init__(self):
        super(SpeedController, self).__init__()
        self.speed_up = 1.005
        self.slow_down = 0.995

    def update_vehicles(self):
        bikes = []
        for vehicle_id in self.activeVehicles.keys():
            if "bike" in vehicle_id:
                bikes.append(vehicle_id)

        # set speed
        if check_for_keyboard_input() == "speed_up":
            for bike_id in bikes:
                current_speed = traci.vehicle.getSpeed(bike_id)
                print("current speed: ", current_speed)
                new_speed = current_speed * self.speed_up
                traci.vehicle.setSpeed(bike_id, new_speed)
        if check_for_keyboard_input() == "slow_down":
            for bike_id in bikes:
                current_speed = traci.vehicle.getSpeed(bike_id)
                print("current speed: ", current_speed)
                new_speed = current_speed * self.slow_down
                traci.vehicle.setSpeed(bike_id, new_speed)
