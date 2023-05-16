import enum

import traci  # noqa

import simulation_classes.Vehicles.vehicleClass as v


class Risk(enum.Enum):
    NoRisk = 0
    SendWarning = 1
    Internvene = 2
    Collision = 3


class CollisionWarningAlgorithm:

    def __init__(self, bike: v.Vehicle):
        self.bike = bike
        self.risk_assessment = {}           # {vehicle_id: Risk}
        self.collision_warning_messages = []

    def get_current_risk(self) -> Risk:
        highest_risk = Risk.NoRisk
        for vehicle_id, risk in self.risk_assessment.items():
            if risk.value > highest_risk.value:
                highest_risk = risk
        return highest_risk
