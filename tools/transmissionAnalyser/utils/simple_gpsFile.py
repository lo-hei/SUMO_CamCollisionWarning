import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


class SimpleGpsFile:

    def __init__(self, file_path):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.file_path = file_path
        self.gps_data = self.load_data()

    def load_data(self):
        print("read simpleGpsFile", self.file_path)

        df = pd.read_csv(self.file_path, sep=",", skipinitialspace=True)

        return df

    def get_positions(self):
        latitude = []
        longitude = []
        time = []

        for i in range(len(self.gps_data.index)):
            lat = self.gps_data.loc[self.gps_data.index[i], "latitude"]
            lon = self.gps_data.loc[self.gps_data.index[i], "longitude"]
            t = self.gps_data.loc[self.gps_data.index[i], "time"]
            latitude.append(lat)
            longitude.append(lon)
            time.append(t)

        return {"latitude": latitude, "longitude": longitude, "time": time}

    def get_starting_time(self):
        times = self.get_positions()["time"]
        for t in times:
            if t > 600000000:
                return t
        return False

    def cut_by_time(self, starting_time):
        self.gps_data = self.gps_data.drop(self.gps_data[self.gps_data.time < starting_time].index)
