import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


class CamFile:

    def __init__(self, file_path):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.file_path = file_path
        self.cam_data = self.load_data()

    def load_data(self):
        print("read camFile", self.file_path)

        df = pd.read_csv(self.file_path, sep=",", skipinitialspace=True)

        df = df.drop(df[df.receiveSendTime < 600000000000000].index)
        df["receiveSendTime"] = df["receiveSendTime"] / 1000000
        df["latitude"] = df["latitude"] * 0.0000001
        df["longitude"] = df["longitude"] * 0.0000001

        return df

    def get_positions(self):
        latitude = []
        longitude = []

        for i in range(len(self.cam_data.index)):
            lat = self.cam_data.loc[self.cam_data.index[i], "latitude"]
            lon = self.cam_data.loc[self.cam_data.index[i], "longitude"]
            latitude.append(lat)
            longitude.append(lon)

        return latitude, longitude

    def get_ids(self):
        ids = []

        for i in range(len(self.cam_data.index)):
            id = self.cam_data.loc[self.cam_data.index[i], "messageID"]
            ids.append(id)

        return ids

    def get_times(self):
        times = []

        for i in range(len(self.cam_data.index)):
            time = self.cam_data.loc[self.cam_data.index[i], "receiveSendTime"]
            times.append(time)

        return times

    def get_starting_time(self):
        times = self.get_times()
        for t in times:
            if t > 600000000:
                return t

        return False

    def cut_by_time(self, starting_time):
        self.cam_data = self.cam_data.drop(self.cam_data[self.cam_data.receiveSendTime < starting_time].index)

    def get_extended_ids(self):
        ids = []

        for i in range(len(self.cam_data.index)):
            id = self.cam_data.loc[self.cam_data.index[i], "messageID"]
            genDeltaTime = self.cam_data.loc[self.cam_data.index[i], "generationDeltaTime"]
            ids.append(str(str(id) + "_" + str(genDeltaTime)))

        return ids
