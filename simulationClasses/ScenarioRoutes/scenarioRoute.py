import random
import numpy


def get_random_prob_mean_1_or_1():
    if bool(random.getrandbits(1)):
        while True:
            r = 1 - abs(numpy.random.normal(loc=0, scale=0.05))
            if 1 >= r >= 0:
                return r
    else:
        return 1


def get_random_prob_mean_1():
    while True:
        r = 1 - abs(numpy.random.normal(loc=0, scale=0.05))
        if 1 >= r >= 0:
            return r


def get_random_prob_mean_0():
    while True:
        r = abs(numpy.random.normal(loc=0, scale=0.05))
        if 1 >= r >= 0:
            return r


def get_random_change_max_6():
    while True:
        r = abs(numpy.random.normal(loc=0, scale=1.6))
        if 3 >= r >= -3:
            return r


class ScenarioRoute:

    def __init__(self, name, number_bikes, number_cars, repeats, start_bike, start_car):
        self.name = name
        self.number_bikes = number_bikes
        self.number_cars = number_cars
        self.repeats = repeats

        self.__start_bike = start_bike
        self.__start_car = start_car
        self.start_bike = start_bike
        self.start_car = start_car

        self.__jmIgnoreJunctionFoeProb = 1
        self.__jmIgnoreFoeProb = 1
        self.__impatience = 1
        self.jmIgnoreJunctionFoeProb = self.__jmIgnoreJunctionFoeProb
        self.jmIgnoreFoeProb = self.__jmIgnoreFoeProb
        self.impatience = self.__impatience


    def tweak(self):
        # changes starting-parameter by a small amount

        self.jmIgnoreFoeProb = get_random_prob_mean_1_or_1()
        self.jmIgnoreJunctionFoeProb = get_random_prob_mean_1_or_1()
        self.impatience = random.choice([0, 1])

        self.start_bike = self.__start_bike + get_random_change_max_6()
        self.start_car = self.__start_car + get_random_change_max_6()