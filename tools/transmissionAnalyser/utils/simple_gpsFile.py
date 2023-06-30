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
        df = pd.read_csv(self.file_path, skipinitialspace=True)

        # set time_to_gps to the time, where first value-change in coordinates is
        self.time_to_gps = 0
        self.time_to_good_gps = 0
        gps_found = False
        good_gps_found = False
        last_lat = df.iloc[0]['latitude']
        last_lon = df.iloc[0]['longitude']
        first_time = df.iloc[0]['system_time']
        for index, row in df.iterrows():
            if row['system_time'] > 600000000000 and first_time == df.iloc[0]['system_time']:
                self.time_to_gps = df.iloc[index - 1]['system_time'] - first_time
                self.time_to_good_gps = df.iloc[index - 1]['system_time'] - first_time
                first_time = row['system_time']
            if (not last_lat == row['latitude'] or not last_lon == row['longitude']) and not gps_found:
                self.time_to_gps += row['system_time'] - first_time
                gps_found = True
            if gps_found and not good_gps_found:
                if row['err_semi_major'] < 20:
                    print(self.time_to_good_gps, row['system_time'], first_time)
                    self.time_to_good_gps += row['system_time'] - first_time
                    good_gps_found = True
                    break

        if gps_found:
            print(self.file_path.split("\\")[-1], " -- Time to GPS:", self.time_to_gps / 1000000, "s")
        else:
            print(self.file_path.split("\\")[-1], " -- Time to GPS: NOT FOUND")

        if good_gps_found:
            print(self.file_path.split("\\")[-1], " -- Time to Good GPS:", self.time_to_good_gps / 1000000, "s")
        else:
            print(self.file_path.split("\\")[-1], " -- Time to Good GPS: NOT FOUND")

        df = df[df['gps_time'] > 0.0]
        df = df[df['latitude'] > 0.0]
        df = df[df['latitude'].notna()]
        df = df[df['longitude'] > 0.0]
        df = df[df['longitude'].notna()]
        df = df[df['hdop'] < 99]

        print("read simpleGpsFile", self.file_path, "(Length:", len(df.index), ")")

        return df

    def get_positions(self):
        latitude = []
        longitude = []
        time = []

        for i in range(len(self.gps_data.index)):
            lat = self.gps_data.loc[self.gps_data.index[i], "latitude"]
            lon = self.gps_data.loc[self.gps_data.index[i], "longitude"]
            t = self.gps_data.loc[self.gps_data.index[i], "gps_time"]
            latitude.append(lat)
            longitude.append(lon)
            time.append(t)

        return {"latitude": latitude, "longitude": longitude, "gps_time": time}

    def get_starting_time(self):
        times = self.get_positions()["gps_time"]
        for t in times:
            if t > 600000000:
                return t
        return False

    def get_end_time(self):
        times = self.get_positions()["gps_time"]
        times.reverse()
        for t in times:
            if t > 600000000:
                return t

        return False

    def cut_start_by_time(self, starting_time):
        self.gps_data = self.gps_data.drop(self.gps_data[self.gps_data.gps_time < starting_time].index)

    def cut_end_by_time(self, end_time):
        self.gps_data = self.gps_data.drop(self.gps_data[self.gps_data.gps_time > end_time].index)

