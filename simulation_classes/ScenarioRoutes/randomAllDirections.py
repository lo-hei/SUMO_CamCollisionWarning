import random

import simulation_classes.ScenarioRoutes.scenarioRoute as sr


class RandomAllDirections(sr.ScenarioRoute):

    def __init__(self):
        name = "Cars are spawning random from all directions"
        number_bikes = 9999
        number_cars = 9999
        super(RandomAllDirections, self).__init__(name=name, number_bikes=number_bikes, number_cars=number_cars)

    def create_route_file(self):
        random.seed(42)  # make tests reproducible
        N = 10000  # number of time steps
        # demand per second from different directions
        p_WE = 1 / 5
        p_EW = 1 / 5
        p_NS = 1 / 10

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
                <vType id="typeWE_car" accel="2" decel="4.5" sigma="0.8" tau="0.6" length="5" minGap="2.5" maxSpeed="25" \
                guiShape="passenger"/>
                <vType id="typeNS_car" accel="2" decel="4.5" sigma="0.8" tau="0.6" length="5" minGap="2.5" maxSpeed="25" \
                guiShape="passenger"/>

                <vType id="typeWE_bike" accel="1.2" decel="3" sigma="0.8" tau="0.3" length="1.6" minGap="0.5" maxSpeed="10" \
                guiShape="bicycle"/>
                <vType id="typeNS_bike" accel="1.2" decel="3" sigma="0.8" tau="0.3" length="1.6" minGap="0.5" maxSpeed="10" \
                guiShape="bicycle"/>

                <route id="right_straight" edges="51o 1i 2o 52i" />
                <route id="left_straight" edges="52o 2i 1o 51i" />
                <route id="down_straight" edges="54o 4i 3o 53i" />

                <route id="right_turnLeft" edges="51o 1i 4o 54i" />
                <route id="left_turnLeft" edges="52o 2i 3o 53i" />
                <route id="down_turnLeft" edges="54o 4i 2o 52i" />

                """, file=routes)

            vehNr = 0
            for i in range(N):

                if random.uniform(0, 1) < p_WE:
                    # car or bike starting West
                    if random.uniform(0, 1) < 0.5:
                        # car driving WE
                        if random.uniform(0, 1) < 0.5:
                            # car driving straight
                            print('<vehicle id="car_right_%i" type="typeWE_car" route="right_straight" depart="%i" />' %
                                  (vehNr, i), file=routes)
                        else:
                            # car turning left
                            print('<vehicle id="car_right_%i" type="typeWE_car" route="right_turnLeft" depart="%i" />' %
                                  (vehNr, i), file=routes)
                    else:
                        # bike driving WE
                        if random.uniform(0, 1) < 0.5:
                            # bike driving straight
                            print(
                                '<vehicle id="bike_right_%i" type="typeWE_bike" route="right_straight" depart="%i" />' %
                                (vehNr, i), file=routes)
                        else:
                            # bike turning left
                            print(
                                '<vehicle id="bike_right_%i" type="typeWE_bike" route="right_turnLeft" depart="%i" />' %
                                (vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < p_EW:
                    # car or bike starting East
                    if random.uniform(0, 1) < 0.5:
                        # car driving EW
                        if random.uniform(0, 1) < 0.5:
                            # car driving straight
                            print('<vehicle id="car_right_%i" type="typeWE_car" route="down_straight" depart="%i" />' %
                                  (vehNr, i), file=routes)
                        else:
                            # car turning left
                            print('<vehicle id="car_right_%i" type="typeWE_car" route="down_turnLeft" depart="%i" />' %
                                  (vehNr, i), file=routes)
                    else:
                        # bike driving EW
                        if random.uniform(0, 1) < 0.5:
                            # bike driving straight
                            print(
                                '<vehicle id="bike_right_%i" type="typeWE_bike" route="down_straight" depart="%i" />' %
                                (vehNr, i), file=routes)
                        else:
                            # bike turning left
                            print(
                                '<vehicle id="bike_right_%i" type="typeWE_bike" route="down_turnLeft" depart="%i" />' %
                                (vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < p_NS:
                    # car or bike starting North
                    if random.uniform(0, 1) < 0.5:
                        # car driving EW
                        if random.uniform(0, 1) < 0.5:
                            # car driving straight
                            print('<vehicle id="car_right_%i" type="typeNS_car" route="left_straight" depart="%i" />' %
                                  (vehNr, i), file=routes)
                        else:
                            # car turning left
                            print('<vehicle id="car_right_%i" type="typeNS_car" route="left_turnLeft" depart="%i" />' %
                                  (vehNr, i), file=routes)
                    else:
                        # bike driving EW
                        if random.uniform(0, 1) < 0.5:
                            # bike driving straight
                            print(
                                '<vehicle id="bike_right_%i" type="typeNS_bike" route="left_straight" depart="%i" />' %
                                (vehNr, i), file=routes)
                        else:
                            # bike turning left
                            print(
                                '<vehicle id="bike_right_%i" type="typeNS_bike" route="left_turnLeft" depart="%i" />' %
                                (vehNr, i), file=routes)
                    vehNr += 1

            print("</routes>", file=routes)
