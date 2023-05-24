from simulationClasses.ScenarioRoutes.bikeLeftCarStraight import BikeLeftCarStraight
from simulationClasses.ScenarioRoutes.bikeStraightCarStraight import BikeStraightCarStraight


class RouteManager:

    def __init__(self):
        self.number_of_available_scenarios = 2
        self.scenarios_list = []

    def get_bike_left_car_straight(self, repeats):
        scenario = BikeLeftCarStraight(repeats=repeats)
        scenario.tweak()
        scenario.create_route_file()
        return scenario

    def get_bike_straight_car_straight(self, repeats):
        scenario = BikeStraightCarStraight(repeats=repeats)
        scenario.tweak()
        scenario.create_route_file()
        return scenario
