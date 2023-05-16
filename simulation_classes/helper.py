import math
import traci  # noqa


def distance(longitude_1, latitude_1, longitude_2, latitude_2):
    return math.sqrt((longitude_2 - longitude_1)**2 + (latitude_1 - latitude_2)**2)


def color_vehicle_green(vehicle_id):
    traci.vehicle.setColor(vehicle_id, (0, 255, 0, 255))


def color_vehicle_red(vehicle_id):
    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))


def color_vehicle_yellow(vehicle_id):
    traci.vehicle.setColor(vehicle_id, (255, 255, 0, 255))


def color_vehicle_blue(vehicle_id):
    traci.vehicle.setColor(vehicle_id, (0, 0, 255, 255))
