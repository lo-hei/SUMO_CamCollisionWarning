from simulationClasses.ScenarioRoutes.bikeLeftCarStraight import BikeLeftCarStraight
from simulationClasses.ScenarioRoutes.bikeStraightCarStraight import BikeStraightCarStraight
from simulationClasses.ScenarioRoutes.evaluationScenario import EvaluationScenario


class RouteManager:

    def __init__(self):
        self.number_of_available_scenarios = 3
        self.scenarios_list = []
        self.gap_between_repeats = 40

    def get_bike_left_car_straight(self, repeats):
        scenario = BikeLeftCarStraight(repeats=repeats, gap_between_repeats=self.gap_between_repeats)
        scenario.tweak()
        scenario.create_route_file()
        return scenario

    def get_bike_straight_car_straight(self, repeats):
        scenario = BikeStraightCarStraight(repeats=repeats, gap_between_repeats=self.gap_between_repeats)
        scenario.tweak()
        scenario.create_route_file()
        return scenario

    def get_evaluation_scenario(self, repeats):
        scenario = EvaluationScenario(repeats=repeats, gap_between_repeats=self.gap_between_repeats)
        scenario.tweak()
        scenario.create_route_file()
        return scenario
