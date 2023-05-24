import json
import os
from pathlib import Path

import numpy as np


class GpsModel:

    def __init__(self, model_name):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()
        self.model_name = model_name
        self.model_path = self.root_path + "\\models\\stored_GpsModels\\" + model_name + "\\"

        self.heatmap = None
        self.heatmap_size = None
        self.gps_frequency = None  # Hz

    def load_model(self):
        if os.path.isdir(self.model_path):
            heatmap_path = self.model_path + "heatmap.txt"
            self.heatmap = np.loadtxt(heatmap_path)

            parameter_path = self.model_path + "parameter.json"
            with open(parameter_path, "r+") as j:
                parameters = json.load(j)
            self.heatmap_size = parameters['heatmap_size']
            self.gps_frequency = parameters['gps_frequency']

        else:
            print("Model not found. (", self.model_name, ")")
            return None

    def save_model(self):
        if not self.heatmap is None:

            if not self.heatmap_size is None:

                if not self.gps_frequency is None:

                    if not os.path.isdir(self.model_path):
                        os.mkdir(self.model_path)

                    heatmap_path = self.model_path + "heatmap.txt"
                    np.savetxt(heatmap_path, self.heatmap, fmt='%1.6e')

                    parameters = {}
                    parameters['heatmap_size'] = self.heatmap_size
                    parameters['gps_frequency'] = self.gps_frequency

                    parameter_path = self.model_path + "parameter.json"
                    with open(parameter_path, "w+") as j:
                        json.dump(parameters, j)

                    return True

                else:
                    print("gps_frequency not set. Model not saved.")
                    return False
            else:
                print("heatmap_size not set. Model not saved.")
                return False
        else:
            print("heatmap not set. Model not saved.")
            return False

    def apply_inaccuracy(self, coordinates):
        print("No apply_inaccuracy implemented. Normal coordinates are returned")
        return coordinates

