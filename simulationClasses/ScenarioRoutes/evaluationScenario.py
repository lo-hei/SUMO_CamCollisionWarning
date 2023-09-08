import random

import numpy as np

import simulationClasses.ScenarioRoutes.scenarioRoute as sr


class EvaluationScenario(sr.ScenarioRoute):

    def __init__(self, repeats, gap_between_repeats):
        name = "Evaluation Scenario, where Bike is coming from one direction " \
               "and car is coming from either left or right (seen from the bike)"

        number_bikes = 1
        number_cars = 1

        start_bike = 0
        start_car = 30

        super(EvaluationScenario, self).__init__(name=name, number_bikes=number_bikes, number_cars=number_cars,
                                                 repeats=repeats, start_bike=start_bike, start_car=start_car)

        self.gap_between_repeats = gap_between_repeats
        self.statistic_routes = {}

    def create_route_file(self):
        with open("simulationData/cross.rou.xml", "w") as routes:
            print("<routes>", file=routes)

            for r in range(self.repeats):
                print(
                    "<vType id='type_car_%i' accel='2' decel='4.5' sigma='0.8' tau='0.6' length='5' minGap='2.5' maxSpeed='12' "
                    "jmIgnoreFoeProb='%i' jmIgnoreJunctionFoeProb='%i' impatience='%i' guiShape='passenger'/>"
                    % (r, self.jmIgnoreFoeProb, self.jmIgnoreJunctionFoeProb, self.impatience),
                    file=routes)

                print(
                    "<vType id='type_bike_%i' accel='1.2' decel='3' sigma='0.8' tau='0.3' length='1.6' minGap='1' maxSpeed='2.56' "
                    "jmIgnoreFoeProb='%i' jmIgnoreJunctionFoeProb='%i' impatience='%i' guiShape='bicycle'/>"
                    % (r, self.jmIgnoreFoeProb, self.jmIgnoreJunctionFoeProb, self.impatience),
                    file=routes)

                # self.tweak()

            print("<route id='left_straight' edges='51o 1i 2o 52i' />", file=routes)
            print("<route id='left_turnLeft' edges='51o 1i 3o 53i' />", file=routes)
            print("<route id='left_turnRight' edges='51o 1i 4o 54i' />", file=routes)

            print("<route id='right_straight' edges='52o 2i 1o 51i' />", file=routes)
            print("<route id='right_turnLeft' edges='52o 2i 3o 53i' />", file=routes)
            print("<route id='right_turnRight' edges='52o 2i 4o 54i' />", file=routes)

            print("<route id='up_straight' edges='54o 4i 3o 53i' />", file=routes)
            print("<route id='up_turnLeft' edges='54o 4i 2o 52i' />", file=routes)
            print("<route id='up_turnRight' edges='54o 4i 1o 51i' />", file=routes)

            print("<route id='down_straight' edges='53o 3i 4o 54i' />", file=routes)
            print("<route id='down_turnLeft' edges='53o 3i 1o 51i' />", file=routes)
            print("<route id='down_turnRight' edges='53o 3i 2o 52i' />", file=routes)

            for r in range(self.repeats):

                random_bike_path = random.choices([0, 1, 2, 3], weights=[1, 1, 1, 1], k=1)[0]
                random_car_left_or_right = random.choices([0, 1], weights=[1, 1], k=1)[0]

                if random_bike_path == 0:
                    # bike is going straight
                    print('<vehicle id="bike_straight_%i" type="type_bike_%i" route="left_straight" depart="%i" />' %
                          (r, r, self.start_bike + self.gap_between_repeats * r), file=routes)
                    if random_car_left_or_right == 0:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="down_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)
                    elif random_car_left_or_right == 1:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="up_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)

                elif random_bike_path == 1:
                    # bike is going straight
                    print('<vehicle id="bike_straight_%i" type="type_bike_%i" route="right_straight" depart="%i" />' %
                          (r, r, self.start_bike + self.gap_between_repeats * r), file=routes)
                    if random_car_left_or_right == 0:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="down_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)
                    elif random_car_left_or_right == 1:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="up_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)

                elif random_bike_path == 2:
                    # bike is going straight
                    print('<vehicle id="bike_straight_%i" type="type_bike_%i" route="up_straight" depart="%i" />' %
                          (r, r, self.start_bike + self.gap_between_repeats * r), file=routes)
                    if random_car_left_or_right == 0:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="left_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)
                    elif random_car_left_or_right == 1:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="right_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)

                elif random_bike_path == 3:
                    # bike is going straight
                    print('<vehicle id="bike_straight_%i" type="type_bike_%i" route="down_straight" depart="%i" />' %
                          (r, r, self.start_bike + self.gap_between_repeats * r), file=routes)
                    if random_car_left_or_right == 0:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="left_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)
                    elif random_car_left_or_right == 1:
                        print('<vehicle id="car_straight_%i" type="type_car_%i" route="right_straight" depart="%i" />' %
                              (r, r, self.start_car + self.gap_between_repeats * r), file=routes)

                self.tweak()

            print("</routes>", file=routes)
