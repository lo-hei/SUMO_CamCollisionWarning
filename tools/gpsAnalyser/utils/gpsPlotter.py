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

    def plot_gps_parameter(self, file_names, interval=None, type="speed_heading"):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), gridspec_kw={'height_ratios': [2, 2]}, dpi=200)


        file_1 = self.file_plotter[file_names[0]]
        file_2 = self.file_plotter[file_names[1]]

        if type == "speed_heading":
            speed_1 = file_1.get_speed()
            speed_2 = file_2.get_speed()
            heading_1 = file_1.get_heading()
            heading_2 = file_2.get_heading()

            time_1 = file_1.get_times()
            time_2 = file_2.get_times()

            if not interval is None:
                speed_1 = speed_1[interval[0]:interval[1]]
                speed_2 = speed_2[interval[0]:interval[1]]
                heading_1 = heading_1[interval[0]:interval[1]]
                heading_2 = heading_2[interval[0]:interval[1]]
                time_1 = time_1[interval[0]:interval[1]]
                time_2 = time_2[interval[0]:interval[1]]

            ax[0].plot(time_1, speed_1, label="Geschwindigkeit (GPS am Tretlager)", color="darkblue")
            ax[0].plot(time_2, speed_2, label="Geschwindigkeit (GPS am Rahmen)", color="darkorange")
            ax[0].set_ylim([-0.5, 12])
            ax[1].plot(heading_1, label="GPS_extern")
            ax[1].plot(heading_2, label="GPS_intern")

            ax[0].legend()
            ax[1].legend()

        if type == "altitude":
            altitude_1, altitude_error_1 = file_1.get_altitude_and_error()
            altitude_2, altitude_error_2 = file_2.get_altitude_and_error()

            time_1 = file_1.get_times()
            time_2 = file_2.get_times()

            if not interval is None:
                altitude_1 = altitude_1[interval[0]:interval[1]]
                altitude_error_1 = altitude_error_1[interval[0]:interval[1]]
                altitude_2 = altitude_2[interval[0]:interval[1]]
                altitude_error_2 = altitude_error_2[interval[0]:interval[1]]
                time_1 = time_1[interval[0]:interval[1]]
                time_2 = time_2[interval[0]:interval[1]]

            ax[0].plot(time_1, altitude_1, label="GPS_extern")
            ax[0].plot(time_2, altitude_2, label="GPS_intern")
            ax[1].plot(time_1, altitude_error_1, label="GPS_extern")
            ax[1].plot(time_2, altitude_error_2, label="GPS_intern")

            ax[0].legend()
            ax[1].legend()

        if type == "error":
            major_1, minor_1, orientation_1 = file_1.get_errors()
            major_2, minor_2, orientation_2 = file_2.get_errors()

            speed_1 = file_1.get_speed()

            time_1 = file_1.get_times()
            time_2 = file_2.get_times()

            if not interval is None:
                major_1 = major_1[interval[0]:interval[1]]
                minor_1 = minor_1[interval[0]:interval[1]]
                major_2 = major_2[interval[0]:interval[1]]
                minor_2 = minor_2[interval[0]:interval[1]]
                time_1 = time_1[interval[0]:interval[1]]
                time_2 = time_2[interval[0]:interval[1]]
                speed_1 = speed_1[interval[0]:interval[1]]

            ax_2 = ax[0].twinx()
            ax[0].plot(speed_1, label="GPS Geschwindigkeit", color="orange")
            ax[0].plot([], [], label="Major GPS-Error", color="darkblue")
            # ax[0].plot(time_2, major_2, label="GPS_intern")
            ax_2.plot(major_1, label="Major GPS-Error", color="darkblue")
            # ax[1].plot(time_2, minor_2, label="GPS_intern")

            ax[0].legend(loc="upper left")
            ax[1].legend()

        if type == "hdop":
            hdop_1 = file_1.get_hdop()
            hdop_2 = file_2.get_hdop()

            time_1 = file_1.get_times()
            time_2 = file_2.get_times()

            if not interval is None:
                hdop_1 = hdop_1[interval[0]:interval[1]]
                hdop_2 = hdop_2[interval[0]:interval[1]]
                time_1 = time_1[interval[0]:interval[1]]
                time_2 = time_2[interval[0]:interval[1]]

            ax[0].plot(time_1, hdop_1, label="GPS_extern")
            ax[1].plot(time_2, hdop_2, label="GPS_intern")

            ax[0].legend()
            ax[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_gps_track_interpolation(self, file_names, dots, interpolation=True, connection=True, interval=None):
        plt.figure(figsize=(5, 3))
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

            if connection:
                plt.plot(long, lat, color=color_dots[i], alpha=0.5)
            if dots:
                plt.scatter(long, lat, s=15, color=color_dots[i])


        plt.ylim(min(lat) - 0.00001, max(lat) + 0.00001)
        plt.xlim(min(long) - 0.00001, max(long) + 0.00001)
        plt.legend()
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.tight_layout()
        plt.show()

    def plot_gps_error(self, file_name, interval):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), gridspec_kw={'height_ratios': [4, 2, 1]})

        file = self.file_plotter[file_name]

        coord, errors_time, errors = file.get_coordinates_with_error()

        # correct time
        time_start = errors_time[0]
        errors_time = [e - time_start for e in errors_time]

        if not interval is None:
            coord = coord[:interval[1]]
            coord = coord[interval[0]:]

            errors_time = errors_time[:interval[1]]
            errors_time = errors_time[interval[0]:]

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
            major_ax = err_semi_major[i] * 2
            minor_ax = err_semi_minor[i] * 2
            angle_deg = err_major_orientation[i]

            ellipse_patch = mpatches.Ellipse(center, major_ax, minor_ax, angle_deg,
                                             fc='orange', alpha=0.2, ls='solid', ec='orange', lw=3.)

            ax[0].add_patch(ellipse_patch)

        ax[0].scatter(long, lat, s=15)

        ax[0].set_ylim(min(lat) - 10, max(lat) + 10)
        ax[0].set_xlim(min(long) - 10, max(long) + 10)
        ax[0].grid()

        ax1_twimx = ax[1].twinx()
        bins = np.arange(0, 8, 0.2)

        _, bins, patches = ax[1].hist(np.clip(err_semi_major, bins[0], bins[-1]), bins=bins, alpha=0.8, color="orange")

        patches[-1].set_facecolor('red')

        xlabels = np.arange(0, 8, 1).astype(str)
        xlabels[-1] += '+'

        N_labels = len(xlabels)
        # ax[1].set_title(str("Time to first GPS-fix:" + file.time_to_gps))
        ax[1].set_xticks(np.arange(N_labels))
        ax[1].set_xticklabels(xlabels)
        ax[1].set_xlim([0, 8])
        ax[1].set_xlabel("GPS-Error in Meter")
        ax[1].set_ylabel("Anzahl GPS-Punkte \n mit dem Fehler")

        density = gaussian_kde(err_semi_major)
        xs = np.linspace(0, 20, 200)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        ax1_twimx.plot(xs, density(xs), color="darkblue")
        ax1_twimx.set_ylabel("Anteil GPS-Punkte \n mit dem Fehler")

        over_20 = len([e for e in err_semi_major if e > 20])
        print(len(errors_time), errors_time)
        print(len(err_semi_major), err_semi_major)
        ax[2].plot(errors_time, err_semi_major)
        ax[2].set_ylim([0, 20])
        if interval:
            ax[2].set_xlim(interval)
        ax[2].set_ylabel("GPS-Error in Meter")
        ax[2].set_xlabel("Time in seconds")
        ax[2].axvspan(0, over_20, alpha=0.5, color='red')
        avg_err_semi_major = sum(err_semi_major) / len(err_semi_major)
        # ax[2].axhline(avg_err_semi_major, linestyle='--', label=str("average error = " + str(round(avg_err_semi_major, 4))))
        # ax[2].legend()

        plt.tight_layout()
        plt.show()

    def plot_gps_error_v2(self, file_name, file_plotter_list, plot_seconds=120):
        fig = plt.figure(figsize=(5, 2.5), dpi=200)

        min_x_time = 0
        to_plot_time = []
        to_plot_error = []

        for i, file_plotter in enumerate(file_plotter_list):

            file = file_plotter[file_name]

            coord, errors_time, errors = file.get_coordinates_with_error()

            if len(errors_time) == 0 or len(errors) == 0:
                continue

            # correct time
            time_start = errors_time[0]
            errors_time = [e - time_start for e in errors_time]

            if not coord:
                print(file_name, "-- No coordinates found")
                return None

            err_major_orientation, err_semi_major, err_semi_minor = zip(*errors)

            if (min_x_time > errors_time[-1] and errors_time[-1] > 100) or (min_x_time == 0):
                min_x_time = errors_time[-1]

            to_plot_time.append(errors_time)
            to_plot_error.append(err_semi_major)

        min_x_time = 120

        for i, time, error in zip(list(range(len(to_plot_time))), to_plot_time, to_plot_error):
            over_20 = len([e for e in error if e > 20])
            pos_error_avg = int((len(error[over_20:]) / 2) + over_20)
            if len(error[pos_error_avg:]) == 0:
                avg_err_semi_major = sum(error[-10:]) / len(error[-10:])
            else:
                avg_err_semi_major = sum(error[pos_error_avg:]) / len(error[pos_error_avg:])
            plt.axhline(avg_err_semi_major, xmin=0.8, linestyle='--', c=str("C" + str(i)),
                        label=str(r'$\varnothing$ Error Test ' + str(i + 1) + ' = ' + str(round(avg_err_semi_major, 4))))

        for i, time, error in zip(list(range(len(to_plot_time))), to_plot_time, to_plot_error):
            over_20 = len([e for e in error if e > 20])
            plt.axvspan(0, over_20, alpha=0.1, color='red')

            start_alpha = len([i for i in time if i < math.floor(min_x_time * 0.78)])
            plt.plot(time[:start_alpha], error[:start_alpha], c=str("C" + str(i)), alpha=1)
            plt.plot(time[start_alpha-1:], error[start_alpha-1:], c=str("C" + str(i)), alpha=0.4)

        plt.xlim([0, min_x_time])
        plt.ylim([0, 25])
        plt.ylabel("GPS-Error in Meter")
        plt.xlabel("Zeit in Sekunden")
        plt.legend(loc="upper right")

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

    def plot_avg_error_with_time_to_gps(self):
        scenarios = ["perfekt, erschwert, Innenstadt, schlecht"]

        # all values from evaluation-charts
        # avg is calculated by hand and calculator and rounded to 2 digits

        found_gps_perfekt = [0, 0, 0, 0]
        time_to_gps_prefekt = [0, 7, 12, 10]
        avg_error_perfect = [0.67, 1.07, 3.11, 9.6]

        found_gps_hard = [0, 0, 20, 27]
        time_to_gps_hard = [0, 3, 35, 5]
        avg_error_hard = [0.88, 1.42, 6.00, 6.64]

        found_gps_city = [0, 10, 20, 20]
        time_to_gps_city = [0, 10, 9, 8]
        avg_error_city = [1.11, 1.65, 5.28, 8.15]

        found_gps_bad = [0, 50, 77, 77]
        time_to_gps_bad = [0, 8, 35, 50]
        avg_error_bad = [1.43, 2.05, 10, 9.2]


        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(111)

        barWidth = 0.2
        br1 = np.arange(4)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        # perfect
        ax1.bar(br1, avg_error_perfect, color='orange', width=barWidth, linewidth=0,
                edgecolor='grey', label='perfekt')

        # hard
        ax1.bar(br2, avg_error_hard, color='darkorange', width=barWidth, linewidth=0,
                edgecolor='grey', label='erschwert')

        # city
        ax1.bar(br3, avg_error_city, color='red', width=barWidth, linewidth=0,
                edgecolor='grey', label='Innenstadt')

        # bad
        ax1.bar(br4, avg_error_bad, color='darkred', width=barWidth, linewidth=0,
                edgecolor='grey', label='schlecht')

        # Adding Xticks
        plt.xlabel('GPS-Empfänger Position', fontweight='bold', fontsize=10)
        plt.ylabel('Durchschn. GPS_Error in Meter', fontweight='bold', fontsize=8)
        plt.xticks([r + 1.5*barWidth for r in range(4)],
                   ['Referenz', 'Lenker', 'Tretlager', 'Rahmen'])

        plt.legend()
        plt.gca().yaxis.grid(True)
        plt.tight_layout()
        plt.show()



