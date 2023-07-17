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
        self.last_warning = {"time": 0, "warning": Risk.NoRisk}
        self.minimal_duration_warning = 3   # in sec
        self.risk_assessment = {}           # {vehicle_id: Risk}
        self.risk_assessment_history = []   # [[time, risk_assessment], [] ... ]

    def get_current_risk(self) -> Risk:
        highest_risk = Risk.NoRisk
        for vehicle_id, risk in self.risk_assessment.items():
            if risk.value > highest_risk.value:
                highest_risk = risk

        if self.last_warning["warning"].value < highest_risk.value:
            self.last_warning["time"] = self.bike.simulation_manager.time
            self.last_warning["warning"] = highest_risk

        if self.bike.simulation_manager.time - self.last_warning["time"] >= self.minimal_duration_warning:
            self.last_warning["time"] = self.bike.simulation_manager.time
            self.last_warning["warning"] = highest_risk

        return self.last_warning["warning"]
