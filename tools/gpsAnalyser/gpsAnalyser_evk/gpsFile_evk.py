import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from bisect import bisect, bisect_left, bisect_right


class GpsFile:

    def __init__(self, file_path):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.file_path = file_path
        self.output_path = self.root_path + "\\src\\maps"

        self.gps_data = self.load_data()
        self.time_to_gps = None
        self.time_to_good_gps = None

        self.map_path = self.root_path + "\\src\\maps\\" + "canyon_hq_map.png"
        self.map_points = [50.3674, 7.5679, 50.3626, 7.5827]

        image = Image.open(self.map_path, 'r')
        self.map_size = [image.size[0], image.size[1]]

    def load_data(self):
        df = pd.read_csv(self.file_path, skipinitialspace=True)

        # set time_to_gps to the time, where first value-change in coordinates is
        last_lat = df.iloc[0]['latitude']
        last_lon = df.iloc[0]['longitude']
        first_time = df.iloc[0]['system_time']
        for index, row in df.iterrows():
            if not last_lat == row['latitude'] or last_lon == row['longitude']:
                self.time_to_gps = row['system_time'] - first_time

        df = df[df['gps_time'] > 0.0]
        df = df[df['latitude'] > 0.0]
        df = df[df['latitude'].notna()]
        df = df[df['longitude'] > 0.0]
        df = df[df['longitude'].notna()]

        return df

    def get_coordinates(self):
        coordinates = tuple(zip(self.gps_data['latitude'].values, self.gps_data['longitude'].values))
        times = self.gps_data['gps_time'].values

        return coordinates, times

    def get_coordinates_with_error(self):
        errors = tuple(zip(self.gps_data['err_major'].values,
                           self.gps_data['err_semi_major'].values,
                           self.gps_data['error_semi_minor'].values))

        coordinates = tuple(zip(self.gps_data['latitude'].values, self.gps_data['longitude'].values))
        times = self.gps_data['gps_time'].values

        return coordinates, times, errors

    def get_image_coordinates(self):
        img_points = []
        coordinates, _ = self.get_coordinates()
        for d in coordinates:
            x1, y1 = self.scale_to_img(d, (self.map_size[0], self.map_size[1]))
            img_points.append((x1, y1))
        return img_points

    def scale_to_img(self, lat_lon, h_w):
        """
        Conversion from latitude and longitude to the image pixels.
        It is used for drawing the GPS records on the map image.
        :param lat_lon: GPS record to draw (lat1, lon1).
        :param h_w: Size of the map image (w, h).
        :return: Tuple containing x and y coordinates to draw on map image.
        """
        # https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445
        old = (self.map_points[2], self.map_points[0])
        new = (0, h_w[1])
        y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        old = (self.map_points[1], self.map_points[3])
        new = (0, h_w[0])
        x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        # y must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        return int(x), h_w[1] - int(y)

    def get_interpolated_coordinates(self, time_resolution=0.05):
        coordinates, times = self.get_coordinates()
        lat, lon = zip(*coordinates)

        interpolated_times = np.arange(times[0], times[-1], time_resolution)

        lat_new = [lat[0]]
        lon_new = [lon[0]]

        for t in interpolated_times[1:-1]:
            i = bisect_left(times, t)

            left_influence = (t - times[i - 1]) / (times[i] - times[i - 1])
            right_influence = (times[i] - t) / (times[i] - times[i - 1])

            lat_n = lat[i - 1] * left_influence + lat[i] * right_influence
            lon_n = lon[i - 1] * left_influence + lon[i] * right_influence

            lat_new.append(lat_n)
            lon_new.append(lon_n)

        lat_new.append(lat[-1])
        lon_new.append(lon[-1])

        interpolated_coord = list(zip(lat_new, lon_new))
        return interpolated_coord, interpolated_times
