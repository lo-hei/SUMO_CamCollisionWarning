import math
import numpy as np
from simulationClasses.TransmissionModel.transmissionModel import TransmissionModel


class SimpleTransmissionModel(TransmissionModel):

    def __init__(self, model_name):
        super(SimpleTransmissionModel, self).__init__(model_name)

    def apply_uncertainty_delivery(self, distance):

        keys_dist = list(self.transmission_accuracy.keys())
        keys_dist.sort()

        if distance > keys_dist[-1]:
            # if distance is greater than tested range, return FALSE
            return False
        else:
            bin_distance = math.floor(distance / 50) * 50

            if bin_distance in keys_dist:
                prob = self.transmission_accuracy[bin_distance]
            else:
                smaller_bin = [x for x in keys_dist if x < bin_distance][-1]
                bigger_bin = [x for x in keys_dist if x > bin_distance][0]

                smaller_prob = self.transmission_accuracy[smaller_bin]
                bigger_prob = self.transmission_accuracy[bigger_bin]
                prob = (smaller_prob + bigger_prob) / 2

            if np.random.uniform(0, 1, 1) > prob:
                return False
            else:
                return True



