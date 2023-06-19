import enum

import traci  # noqa

import simulationClasses.Vehicles.vehicleClass as v


class Risk(enum.Enum):
    NoRisk = 0
    Warning = 1
    Collision = 2


class CollisionWarningAlgorithm:

    def __init__(self, bike: v.Vehicle):
        self.bike = bike
        self.warning_status = Risk.NoRisk
        self.risk_assessment = {}           # {vehicle_id: Risk}
        self.risk_assessment_history = []   # [[time, risk_assessment], [] ... ]

    def get_current_risk(self) -> Risk:
        highest_risk = Risk.NoRisk
        for vehicle_id, risk in self.risk_assessment.items():
            if risk.value > highest_risk.value:
                highest_risk = risk
        return highest_risk
