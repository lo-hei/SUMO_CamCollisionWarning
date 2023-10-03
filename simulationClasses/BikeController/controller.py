"""
Class to change attributes during runtime
"""

class Controller():

    def __init__(self):
        self.activeVehicles = {}

    def set_active_vehicles(self, active_vehicles):
        self.activeVehicles = active_vehicles