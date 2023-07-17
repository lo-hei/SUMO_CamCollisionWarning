import math
import random

from tools.gpsAnalyser.utils.gpsAnalyser import GpsAnalyser
from simulationClasses.GpsModel.gpsModel import GpsModel
from tools.gpsAnalyser.utils.helper import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal

from tools.gpsAnalyser.utils.orderPoints import order_points_neighbors


class GpsModelTool(GpsAnalyser):

    def __init__(self, gps_cam_log_name, available_files, file_plotter, baseline_file):
        super(GpsModelTool, self).__init__(gps_cam_log_name, available_files, file_plotter)
        self.baseline = baseline_file
        self.model = None

        self.plot_xlim = [-20, 20]
        self.plot_ylim = [-20, 20]

    def create_model(self, gps_file, model_name):
        gps_model = GpsModel(model_name=model_name)
        self.model = gps_model
        heatmap, heatmap_size = self.generate_heatmap(gps_file_name=gps_file, plot=False)

        gps_model.heatmap = heatmap
        gps_model.heatmap_size = heatmap_size
        gps_model.gps_frequency = self.detect_gps_frequency(gps_file_name=gps_file, bin_size=0.1, plot=False)

        self.analyse_error(gps_file_name=gps_file)
        self.model.save_model()
        print("GPS Model saved!")

    def use_model(self, model_name, seconds_to_simulate):

        if self.model is None:
            gps_model = GpsModel(model_name=model_name)
            gps_model.load_model()
            self.model = gps_model

        # use baseline-file for "real path"
        # take random seconds_to_simulate seconds from baseline file
        # generate timestamps with gps_frequency
        fixes = self.get_fixes_from_real_path(seconds_to_simulate)
        # generate error (orientation, major, minor) for every timestamp
        errors = self.simulate_error(n=len(fixes))
        # generate deviation for every timestamp
        deviation = self.generate_deviation_points(len(fixes), errors, plot=False)
        # smooth deviation
        smoothed_deviation = self.smooth_deviation(deviation)
        # add deviation to fixes
        gps_path = self.shift_deviation(fixes, smoothed_deviation)
        # smooth with moving average
        # gps_path_smoothed = self.smooth_gps_path(gps_path, plot=True)

        self.plot_model_gps(gps_path, errors)

    def get_fixes_from_real_path(self, seconds_to_simulate):
        time_resolution = 0.1
        baseline_file = self.file_plotter[self.baseline]
        baseline_interpolation, baseline_interpolation_times = \
            baseline_file.get_interpolated_coordinates(time_resolution=time_resolution)

        fixes = []
        random_start = random.randrange(0, len(baseline_interpolation_times) - math.floor(seconds_to_simulate / time_resolution))
        current_s = 0

        while current_s <= seconds_to_simulate:
            lat = baseline_interpolation[random_start][0]
            lon = baseline_interpolation[random_start][1]
            time = baseline_interpolation_times[random_start]
            fixes.append([lat, lon, time])
            random_timestep = random.choices(list(self.model.gps_frequency.keys()),
                                             weights=list(self.model.gps_frequency.values()), k=1)[0]
            current_s += random_timestep
            random_start += math.ceil(random_timestep / time_resolution)

        return fixes

    def analyse_error(self, gps_file_name):
        LONGTIME = 20

        gps_file = self.file_plotter[gps_file_name]
        _, _, e = gps_file.get_coordinates_with_error()

        errors_orientation, errors_major, errors_minor = zip(*e)

        half = math.ceil(len(errors_major) / 2)
        avg_second_half = sum(errors_major[half:]) / len(errors_major[half:])

        for major_minor, errors in zip(["major", "minor"], [errors_major, errors_minor]):
            start_i = 0
            while (errors[start_i] > avg_second_half * 1.4 and start_i < half) or errors[start_i] == 0:
                start_i += 1

            analyse_errors = errors[start_i:]

            changes = []
            longtime_changes = []

            for i in range(len(analyse_errors) - 1):
                changes.append(analyse_errors[i + 1] - analyse_errors[i])

            for i in range(len(analyse_errors) - LONGTIME):
                longtime_changes.append(analyse_errors[i + LONGTIME] - analyse_errors[i])

            mean_error = sum(analyse_errors) / len(analyse_errors)
            min_error = min(analyse_errors) * 0.9
            max_error = max(analyse_errors) * 1.1

            if major_minor == "major":
                self.model.mean_error_major = mean_error
                self.model.min_error_major = min_error
                self.model.max_error_major = max_error
            elif major_minor == "minor":
                self.model.mean_error_minor = mean_error
                self.model.min_error_minor = min_error
                self.model.max_error_minor = max_error

        prob_change = (len(changes) - changes.count(0)) / len(changes)

        not_zero_changes = [round(e, 1) for e in changes if not e == 0]
        distinct_values_changes = list(set(not_zero_changes))
        not_zero_changes_longtime = [round(e, 1) for e in longtime_changes if not e == 0]
        distinct_values_longtime_changes = list(set(not_zero_changes_longtime))

        changes_probs = {}
        for v in distinct_values_changes:
            v_count = not_zero_changes.count(v)
            v_prob = v_count / len(not_zero_changes)
            changes_probs[v] = round(v_prob, 3)

        longtime_changes_probs = {}
        for v in distinct_values_longtime_changes:
            v_count = not_zero_changes_longtime.count(v)
            v_prob = v_count / len(not_zero_changes_longtime)
            longtime_changes_probs[v] = round(v_prob, 3)

        self.model.prob_change = prob_change
        self.model.changes_probs = changes_probs
        self.model.longtime_changes_probs = longtime_changes_probs

        mean_error_orientation = sum(errors_orientation) / len(errors_orientation)
        self.model.error_orientation = mean_error_orientation

    def simulate_error(self, n):
        LONGTIME = 20

        for major_minor in ["major", "minor"]:

            if major_minor == "major":
                mean_error = self.model.mean_error_major
                min_error = self.model.min_error_major
                max_error = self.model.max_error_major
            elif major_minor == "minor":
                mean_error = self.model.mean_error_minor
                min_error = self.model.min_error_minor
                max_error = self.model.max_error_minor

            sim_changes = []
            sim_error = [mean_error]

            for _ in range(n):

                values_changes = list(self.model.changes_probs.keys())
                weights_changes = list(self.model.changes_probs.values())

                if len(sim_changes) >= LONGTIME:
                    current_longtime_change = sim_changes[-LONGTIME] - sim_changes[-1]

                    for i, v in enumerate(values_changes):
                        weight = weights_changes[i]

                        add_weights = []
                        for key, value in self.model.longtime_changes_probs.items():
                            diff = abs(current_longtime_change - key)
                            add_weights.append(diff * value)
                        add_weights = [w / len(add_weights) for w in add_weights]

                        rand_influence = random.choices(list(self.model.longtime_changes_probs.keys()), weights=add_weights, k=1)[0]
                        longtime_influence = 0.5 * abs(rand_influence - v)
                        if rand_influence > weight:
                            weights_changes[i] = weight + longtime_influence
                        if rand_influence < weight:
                            weights_changes[i] = weight - longtime_influence

                    if min(weights_changes) < 0:
                        weights_changes = [w + abs(min(weights_changes)) for w in weights_changes]

                # make sure that the change will not bring the current value out of min-max-range
                last_value = sim_error[-1]
                for i, v in enumerate(values_changes):
                    if (last_value + v) < min_error or (last_value + v) > max_error:
                        weights_changes[i] = 0

                if np.random.uniform(0, 1) < self.model.prob_change:
                    change = random.choices(values_changes, weights=weights_changes, k=1)[0]
                else:
                    change = 0

                sim_changes.append(change)
                sim_error.append(sim_error[-1] + change)

            if major_minor == "major":
                sim_major_error = sim_error
            elif major_minor == "minor":
                sim_minor_error = sim_error

        sim_orientation_error = []

        VARIANZ_IN_ORIENTATION = 10
        for _ in range(n):
            sim_orientation_error.append(np.random.normal(self.model.error_orientation, VARIANZ_IN_ORIENTATION))

        sim_error = list(zip(sim_orientation_error, sim_major_error, sim_minor_error))
        return sim_error

    def generate_heatmap(self, gps_file_name, plot=True):
        SIGMA = 2
        BINS = 200
        ADD_FACTOR = 1.2

        baseline_file = self.file_plotter[self.baseline]
        comparison_file = self.file_plotter[gps_file_name]

        comparison_coord, comparison_times = comparison_file.get_coordinates()

        baseline_interpolation, baseline_interpolation_times, baseline_interpolation_errors = \
            baseline_file.get_interpolated_coordinates(time_resolution=0.5, return_errors=True)

        distances = []
        deviation_points = []

        for i_compare in range(len(comparison_coord)):
            time = comparison_times[i_compare]
            coord = comparison_coord[i_compare]

            # matching by time
            closest_index = closest_value_index(baseline_interpolation_times, time)

            # matching by distance
            # closest_index = closest_coord_index(baseline_interpolation, coord)

            if not closest_index is None:
                coord_b = baseline_interpolation[closest_index]

                d = distance_earth(coord[0], coord[1], coord_b[0], coord_b[1])
                d_point = point_distance_shift(coord[0], coord[1], coord_b[0], coord_b[1])

                d_error = baseline_interpolation_errors[closest_index]
                d_error_orientation = d_error[0]
                d_error_major = d_error[1]
                d_error_minor = d_error[2]

                a_major = math.sin(d_error_orientation) * d_error_major
                b_major = math.cos(d_error_orientation) * d_error_major

                a_minor = math.sin(d_error_orientation - 90) * d_error_minor
                b_minor = math.cos(d_error_orientation - 90) * d_error_minor

                deviation_points.append(d_point)
                distances.append(d)
                deviation_points.append((d_point[0] + a_major, d_point[1] + b_major))
                distances.append(d)
                deviation_points.append((d_point[0] - a_major, d_point[1] - b_major))
                distances.append(d)
                deviation_points.append((d_point[0] + a_minor, d_point[1] - b_minor))
                distances.append(d)
                deviation_points.append((d_point[0] - a_minor, d_point[1] + b_minor))
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

    def generate_deviation_points(self, points_to_generate, errors, plot=True):
        heatmap_model = self.model.heatmap
        heatmap_model = (heatmap_model - np.min(heatmap_model)) / (np.max(heatmap_model) - np.min(heatmap_model))
        heatmap_size = self.model.heatmap_size

        xlim = (heatmap_size[0], heatmap_size[1])
        ylim = (heatmap_size[2], heatmap_size[3])
        self.plot_xlim = xlim
        self.plot_ylim = ylim
        xres = len(np.sum(heatmap_model, axis=0))
        yres = len(np.sum(heatmap_model, axis=1))

        x_bin_width = (xlim[1] - xlim[0]) / xres
        # 0.4 instead if 0.5 to avoid rounding-errors
        x_coordinates = np.arange(start=xlim[0] + 0.4 * x_bin_width, stop=xlim[1] - 0.4 * x_bin_width, step=x_bin_width)

        y_bin_width = (ylim[1] - ylim[0]) / yres
        # 0.4 instead if 0.5 to avoid rounding-errors
        y_coordinates = np.arange(start=ylim[0] + 0.4 * y_bin_width, stop=ylim[1] - 0.4 * y_bin_width, step=y_bin_width)

        # starting up left and going row by row
        coordinates_list = []
        for y in y_coordinates:
            for x in x_coordinates:
                coordinates_list.append((x, y))

        x = np.linspace(xlim[0], xlim[1], xres)
        y = np.linspace(ylim[0], ylim[1], yres)
        xx, yy = np.meshgrid(x, y)

        d_points = []

        for i in range(points_to_generate):
            center = (0, 0)
            major = errors[i][1]
            minor = errors[i][2]
            angle = errors[i][0]

            phi = (angle / 180) * math.pi - 90
            var_x = major ** 2 * math.cos(phi) ** 2 + minor ** 2 * math.sin(phi) ** 2
            var_y = major ** 2 * math.sin(phi) ** 2 + minor ** 2 * math.cos(phi) ** 2
            cov_xy = (major ** 2 - minor ** 2) * math.sin(phi) * math.cos(phi)

            s1 = [[var_x, cov_xy], [cov_xy, var_y]]
            k1 = multivariate_normal(mean=center, cov=s1)

            # evaluate kernels at grid points
            xxyy = np.c_[xx.ravel(), yy.ravel()]
            zz_1 = k1.pdf(xxyy)

            # reshape
            heatmap_error = zz_1.reshape((xres, yres))
            heatmap_error = (heatmap_error - np.min(heatmap_error)) / (np.max(heatmap_error) - np.min(heatmap_error))

            # combine both heatmaps
            heatmap_combine = np.add(heatmap_error, heatmap_model)

            # select random from combined heatmap
            d_point = random.choices(coordinates_list, weights=heatmap_combine.flatten(), k=100)
            d_points = d_points + d_point

        if plot:
            fig = plt.figure(figsize=(10, 15))
            fig.add_subplot(3, 1, 1)
            plt.imshow(heatmap_error, extent=heatmap_size, origin='lower', cmap="viridis")
            fig.add_subplot(3, 1, 2)
            plt.imshow(heatmap_model, extent=heatmap_size, origin='lower', cmap="viridis")
            fig.add_subplot(3, 1, 3)
            plt.imshow(heatmap_combine, extent=heatmap_size, origin='lower', cmap="viridis")
            plt.tight_layout()
            plt.show()

        if plot:
            plt.figure(figsize=(10, 5))
            x, y = list(zip(*d_points))
            plt.scatter(x, y)
            x, y = list(zip(*d_points))
            plt.scatter(x, y, color="orange")
            plt.xlim(self.plot_xlim)
            plt.ylim(self.plot_ylim)
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()

        return d_points

    def detect_gps_frequency(self, gps_file_name, bin_size=0.1, plot=True):
        comparison_file = self.file_plotter[gps_file_name]
        comparison_coord, comparison_times = comparison_file.get_coordinates()
        time_deltas = [comparison_times[n]-comparison_times[n-1] for n in range(1, len(comparison_times))]

        bins = np.arange(bin_size, max(time_deltas), bin_size)
        gps_frequency = {}

        for i, bin in enumerate(bins):
            gps_frequency[bin] = len([d for d in time_deltas if bin > d >= bin - bin_size]) / len(time_deltas)

        if plot:
            plt.bar(list(gps_frequency.keys()), gps_frequency.values(), width=bin_size*0.8)
            plt.xlim(0, max(gps_frequency.keys()))
            plt.show()

        return gps_frequency

    def shift_deviation(self, gps_points, deviation_m):
        shifted_points = []
        for p, d in zip(gps_points, deviation_m):

            new_lat, new_long = shift_position(p[0], p[1], d[0], d[1])
            shifted_points.append([new_lat, new_long])
        return shifted_points

    def smooth_deviation(self, d_points):
        smoothed_coords = [d_points[0]]

        for i in range(2, len(d_points)):

            if i == 0:
                rand = 0
            elif i == 1 or i == len(d_points) - 1:
                rand = 1
            elif i == 2 or i == len(d_points) - 2:
                rand = np.random.randint(1, 4)
            elif i == 3:
                rand = np.random.randint(1, 4)
            else:
                rand = np.random.randint(1, 5)

            if rand == 0:
                avg_x = d_points[i][0]
                avg_y = d_points[i][1]
            if rand == 1:
                avg_x = (d_points[i][0] + d_points[i - 1][0]) / 2
                avg_y = (d_points[i][1] + d_points[i - 1][1]) / 2
            if rand == 2:
                avg_x = (d_points[i][0] + d_points[i - 1][0] + d_points[i + 1][0]) / 3
                avg_y = (d_points[i][1] + d_points[i - 1][1] + d_points[i + 1][1]) / 3
            if rand == 3:
                avg_x = (d_points[i][0] + d_points[i - 1][0] + d_points[i - 2][0] + d_points[i + 1][0]) / 4
                avg_y = (d_points[i][1] + d_points[i - 1][1] + d_points[i - 2][1] + d_points[i + 1][1]) / 4
            if rand == 4:
                avg_x = (d_points[i][0] + d_points[i - 1][0] + d_points[i - 2][0] + d_points[i + 1][0] + d_points[i + 2][0]) / 5
                avg_y = (d_points[i][1] + d_points[i - 1][1] + d_points[i - 2][1] + d_points[i + 1][1] + d_points[i + 2][1]) / 5

            smoothed_coords.append([avg_x, avg_y])

        return smoothed_coords

    def smooth_gps_path(self, gps_points, plot=False):
        smoothed_coords = [gps_points[0]]

        for i in range(2, len(gps_points)):

            if i == 0:
                rand = 0
            elif i == 1 or i == len(gps_points) - 1:
                rand = 1
            elif i == 2 or i == len(gps_points) - 2:
                rand = np.random.randint(1, 4)
            elif i == 3:
                rand = np.random.randint(1, 4)
            else:
                rand = np.random.randint(1, 5)

            if rand == 0:
                avg_x = gps_points[i][0]
                avg_y = gps_points[i][1]
            if rand == 1:
                avg_x = (gps_points[i][0] + gps_points[i - 1][0]) / 2
                avg_y = (gps_points[i][1] + gps_points[i - 1][1]) / 2
            if rand == 2:
                avg_x = (gps_points[i][0] + gps_points[i - 1][0] + gps_points[i + 1][0]) / 3
                avg_y = (gps_points[i][1] + gps_points[i - 1][1] + gps_points[i + 1][1]) / 3
            if rand == 3:
                avg_x = (gps_points[i][0] + gps_points[i - 1][0] + gps_points[i - 2][0] + gps_points[i + 1][0]) / 4
                avg_y = (gps_points[i][1] + gps_points[i - 1][1] + gps_points[i - 2][1] + gps_points[i + 1][1]) / 4
            if rand == 4:
                avg_x = (gps_points[i][0] + gps_points[i - 1][0] + gps_points[i - 2][0] + gps_points[i + 1][0] + gps_points[i + 2][0]) / 5
                avg_y = (gps_points[i][1] + gps_points[i - 1][1] + gps_points[i - 2][1] + gps_points[i + 1][1] + gps_points[i + 2][1]) / 5

            smoothed_coords.append([avg_x, avg_y])

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(*zip(*gps_points), c="orange", label="not smoothed")
            plt.plot(*zip(*smoothed_coords), c="blue", label="smoothed")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return smoothed_coords

    def plot_model_gps(self, smoothed_gps_points, errors, dots=True):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Plot Baseline
        file = self.file_plotter[self.baseline]

        coord, _ = file.get_coordinates()
        interpolated, _ = file.get_interpolated_coordinates(time_resolution=0.5)

        lat_baseline, long_baseline = zip(*coord)
        lat_baseline_new, long_baseline_new = zip(*interpolated)

        ax[0].plot(long_baseline_new, lat_baseline_new, "-", c="orange")
        if dots:
            ax[0].scatter(long_baseline, lat_baseline, s=15, c="orange")

        # Plot Model-GPS
        lat_gps, lon_gps = zip(*smoothed_gps_points)
        ax[0].plot(lon_gps, lat_gps, "-", c="blue")
        if dots:
            ax[0].scatter(lon_gps, lat_gps, s=15, c="blue")

        major_errors = [e for _, e, _ in errors]
        ax[1].plot(major_errors)
        ax[1].set_ylim([min(major_errors), max(major_errors)])

        plt.tight_layout()
        plt.show()
