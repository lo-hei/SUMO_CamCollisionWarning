from tools.gpsAnalyser.utils.gpsAnalyser import GpsAnalyser
from tools.gpsAnalyser.utils.helper import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde

from PIL import Image, ImageDraw

from tools.gpsAnalyser.utils.helper import coord_mesh_to_meter_mesh


class GpsPlotter(GpsAnalyser):

    def __init__(self, gps_cam_log_name, available_files, file_plotter):
        super(GpsPlotter, self).__init__(gps_cam_log_name, available_files, file_plotter)


    def plot_gps(self, file_names, interval, dots=True):

        plt.figure(figsize=(10, 10))
        color_lines = ["blue", "orange", "green"]
        color_dots = ["darkblue", "darkorange", "darkgreen"]

        for i, file_name in enumerate(file_names):
            file = self.file_plotter[file_name]

            coord, _ = file.get_coordinates()

            if not interval is None:
                coord = coord[:interval[1]]
                coord = coord[interval[0]:]

            lat, long = zip(*coord)

            plt.plot(long, lat, color=color_lines[i])
            if dots:
                pass
                plt.scatter(long, lat, s=15, color=color_dots[i])

        plt.ylim(min(lat) - 0.0001, max(lat) + 0.0001)
        plt.xlim(min(long) - 0.0001, max(long) + 0.0001)
        plt.tight_layout()
        plt.show()

    def plot_gps_track_on_map(self, file_name, interval):
        gps_file = self.file_plotter[file_name]
        image = Image.open(gps_file.map_path, 'r')  # Load map image.
        img_points = gps_file.get_image_coordinates()

        if interval is not None:
            img_points = img_points[interval[0]:]
            img_points = img_points[:interval[1]]

        draw = ImageDraw.Draw(image)
        draw.line(img_points, fill=(255, 0, 0), width=2)  # Draw converted records to the map image.
        output_path = gps_file.output_path + '\\resultMap.png'
        image.show()
        # image.save(output_path)

    def plot_gps_track_interpolation(self, file_names, dots, interpolation=True, interval=None):
        plt.figure(figsize=(10, 10))
        color_lines = ["blue", "orange", "green"]
        color_dots = ["darkblue", "darkorange", "darkgreen"]

        for i, file_name in enumerate(file_names):
            file = self.file_plotter[file_name]

            coord, _ = file.get_coordinates()

            if not interval is None:
                coord = coord[:interval[1]]
                coord = coord[interval[0]:]

            interpolated, _ = file.get_interpolated_coordinates(time_resolution=0.1)

            lat, long = zip(*coord)
            lat_new, long_new = zip(*interpolated)

            if interpolation:
                print(long_new)
                plt.plot(long_new, lat_new, color=color_lines[i])

            if dots:
                plt.scatter(long, lat, s=15, color=color_dots[i])


        plt.ylim(min(lat) - 0.0001, max(lat) + 0.0001)
        plt.xlim(min(long) - 0.0001, max(long) + 0.0001)
        plt.tight_layout()
        plt.show()

    def plot_gps_error(self, file_name, interval):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), gridspec_kw={'height_ratios': [4, 2, 1]})

        file = self.file_plotter[file_name]

        coord, _, errors = file.get_coordinates_with_error()

        if not interval is None:
            coord = coord[:interval[1]]
            coord = coord[interval[0]:]

            errors = errors[:interval[1]]
            errors = errors[interval[0]:]

        if not coord:
            print(file_name, "-- No coordinates found")
            return None

        lat, long = zip(*coord)
        lat, long = coord_mesh_to_meter_mesh(lat, long)
        err_major_orientation, err_semi_major, err_semi_minor = zip(*errors)

        for i in range(len(err_major_orientation)):
            center = np.array([long[i], lat[i]])
            # divide by 100 for conversion from Centimeter to Meter
            major_ax = err_semi_major[i] / 100
            minor_ax = err_semi_minor[i] / 100
            angle_deg = err_major_orientation[i]

            ellipse_patch = mpatches.Ellipse(center, major_ax, minor_ax, angle_deg,
                                             fc='orange', alpha=0.1, ls='solid', ec='orange', lw=3.)

            ax[0].add_patch(ellipse_patch)

        ax[0].scatter(long, lat, s=15)

        ax[0].set_ylim(min(lat) - 10, max(lat) + 10)
        ax[0].set_xlim(min(long) - 10, max(long) + 10)
        ax[0].grid()

        ax1_twimx = ax[1].twinx()
        bins = np.arange(0, 40, 1)

        _, bins, patches = ax[1].hist(np.clip(err_semi_major, bins[0], bins[-1]), bins=bins, alpha=0.4, color="orange")

        patches[-1].set_facecolor('red')

        xlabels = np.arange(0, 40, 2).astype(str)
        xlabels[-1] += '+'

        N_labels = len(xlabels)
        # ax[1].set_title(str("Time to first GPS-fix:" + file.time_to_gps))
        ax[1].set_xticks(2 * np.arange(N_labels))
        ax[1].set_xticklabels(xlabels)
        ax[1].set_xlim([0, 40])

        density = gaussian_kde(err_semi_major)
        xs = np.linspace(0, 40, 200)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        ax1_twimx.plot(xs, density(xs))

        over_20 = len([e for e in err_semi_major if e > 20])
        ax[2].plot(err_semi_major)
        ax[2].set_ylim([0, 20])
        ax[2].axvspan(0, over_20, alpha=0.5, color='red')
        avg_err_semi_major = sum(err_semi_major) / len(err_semi_major)
        ax[2].axhline(avg_err_semi_major, linestyle='--', label=str("avg error = " + str(round(avg_err_semi_major, 4))))
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    def plot_gps_deviation(self, baseline, comparison, style="histogram", interval=None):
        baseline_file = self.file_plotter[baseline]
        comparison_file = self.file_plotter[comparison]

        baseline_coord, baseline_times = baseline_file.get_coordinates()
        comparison_coord, comparison_times = comparison_file.get_coordinates()

        if interval is not None:
            comparison_coord = comparison_coord[interval[0]:]
            comparison_coord = comparison_coord[:interval[1]]

            comparison_times = comparison_times[interval[0]:]
            comparison_times = comparison_times[:interval[1]]

        baseline_interpolation, baseline_interpolation_times = \
            baseline_file.get_interpolated_coordinates(time_resolution=0.5)

        distances = []
        deviation_points = []
        deltas = []

        for i_compare in range(len(comparison_coord)):
            time = comparison_times[i_compare]
            coord = comparison_coord[i_compare]

            # matching by time
            # closest_index = closest_value_index(baseline_interpolation_times, time)

            # matching by distance
            closest_index = closest_coord_index(baseline_interpolation, coord)

            if not closest_index is None:
                coord_b = baseline_interpolation[closest_index]
                delta = abs(time - baseline_interpolation_times[closest_index])
                deltas.append(delta)

                '''
                plt.plot(coord[1], coord[0], marker="o", markersize=5, color="blue")
                plt.annotate(i_compare, (coord[1] + 0.0001, coord[0]), color="blue")
                plt.plot(coord_b[1], coord_b[0], marker="x", markersize=5, color="orange")
                plt.annotate(i_compare, (coord_b[1] - 0.00030, coord_b[0]), color="orange")
                '''

                d = distance_earth(coord[0], coord[1], coord_b[0], coord_b[1])

                d_point = point_distance_shift(coord[0], coord[1], coord_b[0], coord_b[1])
                deviation_points.append(d_point)
                distances.append(d)


        plt.show()

        plt.figure(figsize=(10, 10))

        if style == "histogram":
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3]
            plt.hist(np.clip(distances, bins[0], bins[-1]), bins=bins)
            plt.tight_layout()
            plt.show()

        if style == "drift":
            plt.plot(deviation_points)
            plt.tight_layout()
            plt.show()

        if style == "map":
            y, x = zip(*deviation_points)
            plt.scatter(x, y, c=range(len(x)), cmap="viridis")

            lc = colorline(x, y, cmap='viridis', linewidth=3)
            ax = plt.gca()
            ax.add_collection(lc)

            plt.xlim([-20, 20])
            plt.ylim([-20, 20])
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()

        if style == "heatmap":
            y, x = zip(*deviation_points)

            bins = 1000
            plot_range = [[-5, 5], [-5, 5]]
            sigma = 16

            heatmap, xedges, yedges = np.histogram2d(x, y, range=plot_range, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=sigma)

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            plt.imshow(heatmap.T, extent=extent, origin='lower', cmap="viridis")
            plt.plot(x, y, c="black")

            plt.xlim(plot_range[0])
            plt.ylim(plot_range[1])
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()
