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

        self.map_path = self.root_path + "\\src\\maps\\" + "koblenz.png"
        self.map_points = [50.36791, 7.55336, 50.33801, 7.58683]

        image = Image.open(self.map_path, 'r')
        self.map_size = [image.size[0], image.size[1]]

    def load_data(self):
        file1 = open(self.file_path, 'r')
        Lines = file1.readlines()

        columns = ["time", "lat", "lon", "h_acc", "v_acc"]
        df = pd.DataFrame(columns=columns)

        for line in Lines:
            if "PUBX" in line and "lat" in line and "lon" in line \
                    and "hAcc" in line and "vAcc" in line:
                line_split = line.split(",")

                for split in line_split:

                    if "time" in split:
                        time = split.split("=")[1]
                    if "lat" in split:
                        lat = float(split.split("=")[1])
                    if "lon" in split:
                        lon = float(split.split("=")[1])
                    if "hAcc" in split:
                        h_acc = float(split.split("=")[1])
                    if "vAcc" in split:
                        v_acc = float(split.split("=")[1])

                df.loc[len(df)] = [time, lat, lon, h_acc, v_acc]

        return df

    def get_coordinates(self):
        lat_str = 'lat'
        lon_str = 'lon'

        coordinates = tuple(zip(self.gps_data[lat_str].values, self.gps_data[lon_str].values))
        checked_coordinates = []
        checked_times = []

        times = self.gps_data['time'].values

        for i in range(len(coordinates)):
            if (not coordinates[i] == (0.0, 0.0)) and (not times[i] == 0) and (not str(times[i]) == "nan")\
                    and (not math.isnan(coordinates[i][0])) and (not math.isnan(coordinates[i][1])):
                checked_coordinates.append(coordinates[i])

                # convert time
                current_time_str = str(times[i])
                h_m_s = current_time_str.split(":")
                current_time = 0
                current_time += int(h_m_s[0]) * 60 * 60
                current_time += int(h_m_s[1]) * 60
                current_time += int(h_m_s[2])

                if len(checked_times) > 0:
                    if int(checked_times[-1]) == current_time:
                        if current_time == checked_times[-1]:
                            checked_times[-1] = int(checked_times[-1]) + 0.3
                            current_time += 0.7
                        else:
                            current_time += 0.5

                checked_times.append(current_time)

        checked_coordinates = [x for _, x in sorted(zip(checked_times, checked_coordinates))]
        checked_times = list(sorted(checked_times))

        return checked_coordinates, checked_times

    def get_coordinates_with_error(self):
        errors = tuple(zip([0]*self.gps_data['h_acc'],
                           self.gps_data['h_acc'].values,
                           self.gps_data['h_acc'].values))

        coordinates = tuple(zip(self.gps_data['lat'].values, self.gps_data['lon'].values))
        times = self.gps_data['time'].values

        while coordinates[0] == (0.0, 0.0) or times[0] == 0:
            coordinates = coordinates[1:]
            times = times[1:]
            errors = errors[1:]

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

    def get_interpolated_coordinates(self, time_resolution=0.5):
        coordinates, times = self.get_coordinates()
        lat, lon = zip(*coordinates)

        interpolated_times = np.arange(times[0], times[-1], time_resolution)

        lat_new = [lat[0]]
        lon_new = [lon[0]]

        for t in interpolated_times[1:]:
            i = bisect_left(times, t)
            left_influence = (t - times[i - 1]) / (times[i] - times[i - 1])
            right_influence = (times[i] - t) / (times[i] - times[i - 1])

            lat_n = lat[i - 1] * left_influence + lat[i] * right_influence
            lon_n = lon[i - 1] * left_influence + lon[i] * right_influence

            lat_new.append(lat_n)
            lon_new.append(lon_n)

        interpolated_coord = list(zip(lat_new, lon_new))
        return interpolated_coord, interpolated_times
