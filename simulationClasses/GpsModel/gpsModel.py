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
        self.gps_frequency = {}

        self.mean_error_major = None
        self.min_error_major = None
        self.max_error_major = None

        self.mean_error_minor = None
        self.min_error_minor = None
        self.max_error_minor = None

        self.error_orientation = None

        self.prob_change = None
        self.changes_probs = None
        self.longtime_changes_probs = None

    def load_model(self):
        if os.path.isdir(self.model_path):
            heatmap_path = self.model_path + "heatmap.txt"
            self.heatmap = np.loadtxt(heatmap_path)

            parameter_path = self.model_path + "parameter.json"
            with open(parameter_path, "r+") as j:
                parameters = json.load(j)
            self.heatmap_size = parameters['heatmap_size']

            gps_frequency = parameters['gps_frequency']
            self.changes_probs = {}
            for k, v in gps_frequency.items():
                self.gps_frequency[float(k)] = float(v)

            self.mean_error_major = float(parameters['mean_error_major'])
            self.min_error_major = float(parameters['min_error_major'])
            self.max_error_major = float(parameters['max_error_major'])

            self.mean_error_minor = float(parameters['mean_error_minor'])
            self.min_error_minor = float(parameters['min_error_minor'])
            self.max_error_minor = float(parameters['max_error_minor'])

            self.error_orientation = float(parameters['error_orientation'])
            self.prob_change = float(parameters['prob_change'])

            changes_probs = parameters['changes_probs']
            self.changes_probs = {}
            for k, v in changes_probs.items():
                self.changes_probs[float(k)] = float(v)

            longtime_changes_probs = parameters['longtime_changes_probs']
            self.longtime_changes_probs = {}
            for k, v in longtime_changes_probs.items():
                self.longtime_changes_probs[float(k)] = float(v)

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

                    parameters['mean_error_major'] = self.mean_error_major
                    parameters['min_error_major'] = self.min_error_major
                    parameters['max_error_major'] = self.max_error_major

                    parameters['mean_error_minor'] = self.mean_error_minor
                    parameters['min_error_minor'] = self.min_error_minor
                    parameters['max_error_minor'] = self.max_error_minor

                    parameters['error_orientation'] = self.error_orientation

                    parameters['prob_change'] = self.prob_change
                    parameters['changes_probs'] = self.changes_probs
                    parameters['longtime_changes_probs'] = self.longtime_changes_probs

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

