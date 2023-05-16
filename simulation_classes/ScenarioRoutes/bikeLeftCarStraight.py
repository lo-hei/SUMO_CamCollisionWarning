import random

import simulation_classes.ScenarioRoutes.scenarioRoute as sr


class BikeLeftCarStraight(sr.ScenarioRoute):

    def __init__(self, repeats):
        name = "Bike turn left, car going straight"
        number_bikes = 1
        number_cars = 1

        start_bike = 0
        start_car = 34

        super(BikeLeftCarStraight, self).__init__(name=name, number_bikes=number_bikes, number_cars=number_cars,
                                                  repeats=repeats, start_bike=start_bike, start_car=start_car)

        self.gap_between_repeats = 40

    def create_route_file(self):

        with open("data/cross.rou.xml", "w") as routes:
            print("<routes>", file=routes)

            print("<vType id='type_car' accel='2' decel='4.5' sigma='0.8' tau='0.6' length='5' minGap='2.5' maxSpeed='12' "
                  "jmIgnoreFoeProb='%i' jmIgnoreJunctionFoeProb='%i' impatience='%i' guiShape='passenger'/>"
                  % (self.jmIgnoreFoeProb, self.jmIgnoreJunctionFoeProb, self.impatience),
                  file=routes)

            print("<vType id='type_bike' accel='1.2' decel='3' sigma='0.8' tau='0.3' length='1.6' minGap='1' maxSpeed='2.56' "
                  "jmIgnoreFoeProb='%i' jmIgnoreJunctionFoeProb='%i' impatience='%i' guiShape='bicycle'/>"
                  % (self.jmIgnoreFoeProb, self.jmIgnoreJunctionFoeProb, self.impatience),
                  file=routes)

            print("<route id='down_straight' edges='54o 4i 3o 53i' />", file=routes)
            print("<route id='left_turnLeft' edges='51o 1i 4o 54i' />", file=routes)

            for r in range(self.repeats):
                print('<vehicle id="bike_left_%i" type="type_bike" route="left_turnLeft" depart="%i" />' %
                      (r, self.start_bike + self.gap_between_repeats*r), file=routes)
                print('<vehicle id="car_straight_%i" type="type_car" route="down_straight" depart="%i" />' %
                      (r, self.start_car + self.gap_between_repeats*r), file=routes)
                self.tweak()

            print("</routes>", file=routes)

