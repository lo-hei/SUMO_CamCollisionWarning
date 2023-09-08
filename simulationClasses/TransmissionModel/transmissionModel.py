import json
import os
from pathlib import Path


class TransmissionModel:

    def __init__(self, model_name):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()
        self.model_name = model_name
        self.model_path = self.root_path + "\\models\\stored_TransmissionModels\\" + model_name + "\\"

        self.transmission_time = None
        self.transmission_accuracy = None

    def load_model(self, vehicle_type):
        if os.path.isdir(self.model_path):

            parameter_path = self.model_path + "parameter_" + "bike" + ".json"
            # parameter_path = self.model_path + "parameter_" + str(vehicle_type) + ".json"

            with open(parameter_path, "r+") as j:
                parameters = json.load(j)

            self.transmission_time = parameters['transmission_time']
            transmission_accuracy_keys = parameters['transmission_accuracy_keys']
            transmission_accuracy_values = parameters['transmission_accuracy_values']

            self.transmission_accuracy = {}
            for k, v in zip(transmission_accuracy_keys, transmission_accuracy_values):
                self.transmission_accuracy[float(k)] = float(v)

        else:
            print("Model not found. (", self.model_name, ")")
            return None

    def save_model(self, vehicle_type):
        if not self.transmission_time is None:

            if not self.transmission_accuracy is None:

                if not os.path.isdir(self.model_path):
                    os.mkdir(self.model_path)

                parameters = {}
                parameters['transmission_time'] = float(self.transmission_time)
                parameters['transmission_accuracy_keys'] = list(self.transmission_accuracy.keys())
                parameters['transmission_accuracy_values'] = list(self.transmission_accuracy.values())
                parameter_path = self.model_path + "parameter_" + str(vehicle_type) + ".json"

                with open(parameter_path, "w+") as j:
                    print(parameters, j)
                    json.dump(parameters, j)

                return True

            else:
                print("transmission_accuracy not set. Model not saved.")
                return False
        else:
            print("transmission_time not set. Model not saved.")
            return False

    def apply_inaccuracy(self, distance):
        print("No apply_inaccuracy implemented. Transmission is allowed")
        return True
