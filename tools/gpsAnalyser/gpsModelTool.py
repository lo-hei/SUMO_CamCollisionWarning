import random

from tools.gpsAnalyser.utils.gpsAnalyser import GpsAnalyser
from simulationClasses.GpsModel.gpsModel import GpsModel
from tools.gpsAnalyser.utils.helper import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from tools.gpsAnalyser.utils.orderPoints import order_points_neighbors_duplicates, order_points_neighbors


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

        d_points = self.generate_points(seconds_to_simulate * 2, plot=False)
        d_points_ordered = order_points_neighbors(d_points, plot_xlim=self.plot_xlim, plot_ylim=self.plot_ylim)
        # d_points_ordered = order_points_neighbors_duplicates(d_points, plot_xlim=self.plot_xlim, plot_ylim=self.plot_ylim)
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
