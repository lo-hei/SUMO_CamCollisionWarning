from tools.gpsAnalyser_victor.utils.gpsAnalyser import GpsAnalyser
from tools.gpsAnalyser_victor.utils.helper import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from PIL import Image, ImageDraw


class GpsPlotter(GpsAnalyser):

    def __init__(self, gps_cam_log_name):
        super(GpsPlotter, self).__init__(gps_cam_log_name)

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

    def plot_gps_track_interpolation(self, file_names, dots, interval=None):
        plt.figure(figsize=(10, 10))

        for file_name in file_names:
            print(self.file_plotter)
            file = self.file_plotter[file_name]

            coord, _ = file.get_coordinates()

            if not interval is None:
                coord = coord[:interval[1]]
                coord = coord[interval[0]:]

            print("Start plot", file_name, ":", coord)

            interpolated, _ = file.get_interpolated_coordinates()

            lat, long = zip(*coord)
            lat_new, long_new = zip(*interpolated)

            plt.plot(long_new, lat_new)
            if dots:
                plt.scatter(long, lat, s=15)

        plt.ylim(min(lat) - 0.0001, max(lat) + 0.0001)
        plt.xlim(min(long) - 0.0001, max(long) + 0.0001)
        plt.tight_layout()
        plt.show()

    def plot_gps_deviation(self, baseline, comparison, interval, style="histogram"):
        baseline_file = self.file_plotter[baseline]
        comparison_file = self.file_plotter[comparison]

        baseline_coord, baseline_times = baseline_file.get_coordinates()
        comparison_coord, comparison_times = comparison_file.get_coordinates()

        if interval is not None:
            comparison_coord = comparison_coord[interval[0]:]
            comparison_coord = comparison_coord[:interval[1]]

            comparison_times = comparison_times[interval[0]:]
            comparison_times = comparison_times[:interval[1]]

        print(comparison_coord)
        print(baseline_coord)

        baseline_interpolation, baseline_interpolation_times = \
            baseline_file.get_interpolated_coordinates(time_resolution=0.05)

        distances = []
        deviation_points = []
        deltas = []

        for i_compare in range(len(comparison_coord)):
            time = comparison_times[i_compare]
            coord = comparison_coord[i_compare]

            closest_index = closest_value_index(baseline_interpolation_times, time)

            if not closest_index is None:
                coord_b = baseline_interpolation[closest_index]
                delta = abs(time - baseline_interpolation_times[closest_index])
                deltas.append(delta)

                d = distance_earth(coord[0], coord[1], coord_b[0], coord_b[1])
                d_point = point_distance_shift(coord[0], coord[1], coord_b[0], coord_b[1])
                deviation_points.append(d_point)
                distances.append(d)

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

            xlim = [-20, 20]
            if min(x) < -20:
                xlim[0] = min(x)
            if max(x) > 20:
                xlim[1] = max(x)

            ylim = [-20, 20]
            if min(y) < -20:
                ylim[0] = min(y)
            if max(y) > 20:
                ylim[1] = max(y)

            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()

        if style == "heatmap":
            y, x = zip(*deviation_points)

            bins = 1000
            plot_range = [[-50, 50], [-50, 50]]
            sigma = 16

            heatmap, xedges, yedges = np.histogram2d(x, y, range=plot_range, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=sigma)

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            plt.imshow(heatmap.T, extent=extent, origin='lower', cmap="viridis")
            plt.plot(x, y, c="black")

            plt.xlim([-20, 20])
            plt.ylim([-20, 20])
            plt.hlines(0, -50, 50, colors="black")
            plt.vlines(0, -50, 50, colors="black")
            plt.tight_layout()
            plt.show()