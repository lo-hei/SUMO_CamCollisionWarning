import random

from tools.gpsAnalyser.gpsAnalyser import GpsAnalyser
from tools.gpsAnalyser.gpsModel import GpsModel
from tools.gpsAnalyser.helper import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from PIL import Image, ImageDraw


class GpsModelTool(GpsAnalyser):

    def __init__(self, gps_cam_log_name, baseline_file):
        super(GpsModelTool, self).__init__(gps_cam_log_name)
        self.baseline = baseline_file
        self.model = None

        self.plot_xlim = [-20, 20]
        self.plot_ylim = [-20, 20]

    def create_model(self, gps_file, model_name):
        gps_model = GpsModel(model_name=model_name)
        heatmap, heatmap_size = self.generate_heatmap(gps_file_name=gps_file)

        gps_model.heatmap = heatmap
        gps_model.heatmap_size = heatmap_size
        gps_model.gps_frequency = 10

        self.model = gps_model
        gps_model.save_model()

    def use_model(self, model_name, seconds_to_simulate):

        if self.model is None:
            gps_model = GpsModel(model_name=model_name)
            gps_model.load_model()
            self.model = gps_model

        d_points = self.generate_points(seconds_to_simulate * 2)
        d_points_ordered = self.order_points(d_points)
        self.plot_model_gps(d_points_ordered)



    def generate_heatmap(self, gps_file_name, plot=True):
        SIGMA = 2
        BINS = 100
        ADD_FACTOR = 1.5

        baseline_file = self.file_plotter[self.baseline]
        comparison_file = self.file_plotter[gps_file_name]

        comparison_coord, comparison_times = comparison_file.get_coordinates()

        baseline_interpolation, baseline_interpolation_times = \
            baseline_file.get_interpolated_coordinates(time_resolution=0.05)

        distances = []
        deviation_points = []

        for i_compare in range(len(comparison_coord)):
            time = comparison_times[i_compare]
            coord = comparison_coord[i_compare]

            closest_index = closest_value_index(baseline_interpolation_times, time)

            if not closest_index is None:
                coord_b = baseline_interpolation[closest_index]

                d = distance_earth(coord[0], coord[1], coord_b[0], coord_b[1])
                d_point = point_distance_shift(coord[0], coord[1], coord_b[0], coord_b[1])
                deviation_points.append(d_point)
                distances.append(d)

        # delete leading GPS-Points until Precision is high enought
        avg_distance = sum(distances) / len(distances)
        while distances[0] > 2 * avg_distance:
            distances.pop(0)
            deviation_points.pop(0)

        y, x = zip(*deviation_points)

        plot_range = [[min(x) * ADD_FACTOR, max(x) * ADD_FACTOR],
                      [min(y) * ADD_FACTOR, max(y) * ADD_FACTOR]]

        heatmap, xedges, yedges = np.histogram2d(x, y, range=plot_range, bins=BINS)
        heatmap = gaussian_filter(heatmap, sigma=SIGMA)

        heatmap_size = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        if plot:
            plt.figure(figsize=(10, 10))

            plt.imshow(heatmap.T, extent=heatmap_size, origin='lower', cmap="viridis")

            plt.xlim([min(x) * ADD_FACTOR, max(x) * ADD_FACTOR])
            plt.ylim([min(y) * ADD_FACTOR, max(y) * ADD_FACTOR])
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()

        return heatmap.T, heatmap_size

    def generate_points(self, seconds_to_simulate, plot=True):
        gps_frequency = 10  # Hz
        points_to_generate = seconds_to_simulate * gps_frequency
        heatmap = self.model.heatmap
        heatmap_size = self.model.heatmap_size

        col_sums = np.sum(heatmap, axis=0)
        col_nums = len(col_sums)
        row_sums = np.sum(heatmap, axis=1)
        row_nums = len(row_sums)

        x_start = heatmap_size[0]
        x_end = heatmap_size[1]
        self.plot_xlim = [x_start, x_end]
        x_bin_width = (x_end - x_start) / col_nums
        # 0.4 instead if 0.5 to avoid rounding-errors
        x_coordinates = np.arange(start=x_start + 0.4*x_bin_width, stop=x_end - 0.4*x_bin_width, step=x_bin_width)

        y_start = heatmap_size[2]
        y_end = heatmap_size[3]
        self.plot_ylim = [y_start, y_end]
        y_bin_width = (y_end - y_start) / row_nums
        # 0.4 instead if 0.5 to avoid rounding-errors
        y_coordinates = np.arange(start=y_start + 0.4*y_bin_width, stop=y_end - 0.4*y_bin_width, step=y_bin_width)

        # starting up left and going row by row
        coordinates_list = []
        for y in y_coordinates:
            for x in x_coordinates:
                coordinates_list.append((x, y))

        d_points = random.choices(coordinates_list, weights=heatmap.flatten(), k=points_to_generate)

        if plot:
            plt.figure(figsize=(10, 5))
            x, y = list(zip(*d_points))
            plt.scatter(x, y)
            plt.xlim(self.plot_xlim)
            plt.ylim(self.plot_ylim)
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()

        return d_points

    def order_points(self, d_points, plot=True):
        SHUFFLE_SIZE = 3
        MAXIMAL_DISTANCE = 2

        ordered_points = []
        first_point = random.choice(d_points)
        ordered_points.append(first_point)
        d_points.remove(first_point)

        while len(d_points) > 0:

            # nearest neighbor jump
            last_point = ordered_points[-1]
            distances = []
            for p in d_points:
                d = math.dist(last_point, p)
                distances.append(d)
            min_dist = min(distances)
            current_point_i = distances.index(min_dist)
            current_point = d_points[current_point_i]

            if min_dist < MAXIMAL_DISTANCE:
                print("smaller than MAXIMAL_DISTANCE", min_dist)
                ordered_points.append(current_point)
                d_points.remove(current_point)
            else:
                print("greater than MAXIMAL_DISTANCE", min_dist)
                # find nearest neighbor in already ordered points
                distances_to_ordered = []
                for o in ordered_points:
                    d_o = math.dist(current_point, o)
                    distances_to_ordered.append(d_o)
                min_dist_ordered = min(distances_to_ordered)
                min_dist_i_ordered = distances_to_ordered.index(min_dist_ordered)

                # add point before or after this point
                if min_dist_i_ordered > 0:
                    distance_before = math.dist(current_point, ordered_points[min_dist_i_ordered - 1])
                else:
                    ordered_points.insert(0, current_point)
                    d_points.remove(current_point)
                    continue

                if min_dist_i_ordered < (len(ordered_points) - 2):
                    distance_after = math.dist(current_point, ordered_points[min_dist_i_ordered + 1])
                else:
                    ordered_points.append(current_point)
                    d_points.remove(current_point)
                    continue

                if distance_before < distance_after:
                    # add current_point_i before ordered_points[min_dist_i_ordered]
                    ordered_points.insert(min_dist_i_ordered - 1, current_point)
                    d_points.remove(current_point)
                else:
                    # add current_point_i after ordered_points[min_dist_i_ordered]
                    ordered_points.insert(min_dist_i_ordered + 1, current_point)
                    d_points.remove(current_point)

        if plot:
            plt.figure(figsize=(10, 5))

            x, y = list(zip(*ordered_points))
            plt.scatter(x, y, c=range(len(x)), cmap="viridis")

            lc = colorline(x, y, cmap='viridis', linewidth=3)
            ax = plt.gca()
            ax.add_collection(lc)

            plt.xlim(self.plot_xlim)
            plt.ylim(self.plot_ylim)
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()

        return ordered_points

    def find_neighbors(self, point, d_points, k):
        distances = []
        for p in d_points:
            d = math.dist(point, p)
            distances.append(d)

        points_very_near = sum(i < 5 for i in distances)
        if points_very_near < 10:
            if len(d_points) > points_very_near:
                k = points_very_near
            else:
                k = len(d_points)

        max_distance = max(distances) + 1
        distances_inverse = [(max_distance - d) ** 2 for d in distances]

        neighbors = []
        for _ in range(k):
            if len(d_points) > 0:
                neighbor_i = random.choices(range(len(d_points)), weights=distances_inverse, k=1)[0]

            else:
                return neighbors

            neighbors.append(d_points[neighbor_i])
            del distances_inverse[neighbor_i]
            del d_points[neighbor_i]

        return neighbors

    def plot_model_gps(self, d_points_ordered, dots=True):
        plt.figure(figsize=(10, 10))

        # Plot Baseline
        file = self.file_plotter[self.baseline]

        coord, _ = file.get_coordinates()
        interpolated, _ = file.get_interpolated_coordinates()

        lat_baseline, long_baseline = zip(*coord)
        lat_baseline_new, long_baseline_new = zip(*interpolated)

        plt.plot(long_baseline_new, lat_baseline_new, c="orange")
        if dots:
            plt.scatter(long_baseline, lat_baseline, s=15, c="orange")

        # Plot Model-GPS
        number_model_points = len(d_points_ordered)
        number_interpolated_points = len(interpolated)
        ratio = number_interpolated_points / number_model_points
        nearest_interpolated_points = []
        r = 0
        while r < number_interpolated_points-1:
            nearest_interpolated_points.append(interpolated[math.ceil(r)])
            r += ratio

        lat_model = []
        long_model = []
        for i in range(len(nearest_interpolated_points)):
            new_lat, new_long = shift_position(nearest_interpolated_points[i][0], nearest_interpolated_points[i][1],
                                               d_points_ordered[i][0], d_points_ordered[i][1])
            lat_model.append(new_lat)
            long_model.append(new_long)

        plt.plot(long_model, lat_model)
        if dots:
            plt.scatter(long_model, lat_model, s=15)

        plt.ylim(min(lat_model) - 0.0001, max(lat_model) + 0.0001)
        plt.xlim(min(long_model) - 0.0001, max(long_model) + 0.0001)
        plt.tight_layout()
        plt.show()
