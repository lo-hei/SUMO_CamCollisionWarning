import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.stats import gaussian_kde

from simulationClasses.CollisionWarningAlgorithm.collisionWarningAlgorithm import Risk
from simulationClasses.routeManager import RouteManager


class CwaEvaluator:

    def __init__(self, runs, cwa, evaluation_mode):
        self.distance_collision = 1.5
        self.distance_warning = 5

        self.cwa = cwa
        if type(evaluation_mode) == list:
            self.evaluation_mode = evaluation_mode
        else:
            self.evaluation_mode = [evaluation_mode]

        self.runs = runs

        self.route_manager = RouteManager()

        if evaluation_mode == [7]:
            self.generate_routefile()
            # self.generate_evaluation_routefile()
        else:
            self.generate_routefile()

        # just inactive vehicles from simulationManager
        self.vehicles = None

        self.real_pred_value = []  # [[real value, predicted value], [r_v, p_v], ... ]
        self.warning_time_before_poc = []
        self.collision_time_before_poc = []

    def generate_routefile(self):
        get_bike_left_car_straight = self.route_manager.get_bike_straight_car_straight(repeats=self.runs)

    def generate_evaluation_routefile(self):
        evaluation_scenario = self.route_manager.get_evaluation_scenario(repeats=self.runs)

    def evaluate(self):

        for evaluation_mode in self.evaluation_mode:
            if evaluation_mode == 0:
                continue
            elif evaluation_mode == 1:
                self.plot_vehicle_paths()
            elif evaluation_mode == 2:
                self.plot_distance_between_vehicles(only_cwa=False)
            elif evaluation_mode == 3:
                self.plot_distance_between_vehicles(only_cwa=True)
            elif evaluation_mode == 4:
                self.plot_cwa_parameter()
            elif evaluation_mode == 5:
                self.plot_poc_error()
            elif evaluation_mode == 6:
                self.analyse_cwa_parameter()
            elif evaluation_mode == 7:
                self.evaluate_cwa()
            else:
                print("Wrong evaluation_mode selected")

    def evaluate_cwa(self):
        car_bike_pairs = {}
        for vehicle_id in self.vehicles.keys():
            v_id = vehicle_id.split("_")[-1]
            if v_id in car_bike_pairs.keys():
                if "bike" in vehicle_id:
                    car_bike_pairs[v_id].append(vehicle_id)
                else:
                    car_bike_pairs[v_id].insert(0, vehicle_id)
            else:
                car_bike_pairs[v_id] = [vehicle_id]

        warning_correct = {}
        collision_correct = {}
        warning_but_real_collision = {}
        collision_but_real_warning = {}
        norisk_but_real_warning = {}
        warning_but_real_norisk = {}
        for v_id, (car_id, bike_id) in car_bike_pairs.items():
            car = self.vehicles[car_id]
            bike = self.vehicles[bike_id]

            pred_real, pred_cwa = self.evaluate_bike_car_pair(bike, car)
            print(car_id, bike_id)
            print(pred_cwa, pred_real)

            if pred_real == pred_cwa and pred_real == Risk.Warning:
                warning_correct[v_id] = [car_id, bike_id]
            if pred_real == pred_cwa and pred_real == Risk.Collision:
                collision_correct[v_id] = [car_id, bike_id]
            if pred_real == Risk.Collision and pred_cwa == Risk.Warning:
                warning_but_real_collision[v_id] = [car_id, bike_id]
            if pred_real == Risk.Warning and pred_cwa == Risk.Collision:
                collision_but_real_warning[v_id] = [car_id, bike_id]
            if pred_real == Risk.NoRisk and pred_cwa == Risk.Warning:
                warning_but_real_norisk[v_id] = [car_id, bike_id]
            if pred_real == Risk.Warning and pred_cwa == Risk.NoRisk:
                norisk_but_real_warning[v_id] = [car_id, bike_id]

        self.plot_cwa_prob(car_bike_pairs=warning_correct, title="Warning Correct")
        self.plot_cwa_prob(car_bike_pairs=collision_correct, title="Collision Correct")
        self.plot_cwa_prob(car_bike_pairs=warning_but_real_collision, title="Predicted Warning but real Collision")
        self.plot_cwa_prob(car_bike_pairs=collision_but_real_warning, title="Predicted Collision but real Warning")
        self.plot_cwa_prob(car_bike_pairs=warning_but_real_norisk, title="Prediction Warning but real NoRisk")
        self.plot_cwa_prob(car_bike_pairs=norisk_but_real_warning, title="Prediction NoRisk but real Warning")

        if len(self.warning_time_before_poc) > 0:
            avg_cwa_time_before_warning = sum(self.warning_time_before_poc) / len(self.warning_time_before_poc)
        else:
            avg_cwa_time_before_warning = 0
        if len(self.collision_time_before_poc) > 0:
            avg_cwa_time_before_collision = sum(self.collision_time_before_poc) / len(self.collision_time_before_poc)
        else:
            avg_cwa_time_before_collision = 0

        # --- plot histograms ---
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 1]})
        ax1 = ax[0]
        ax2 = ax[1]

        max_bin = 5 * round(max(max(self.collision_time_before_poc), max(self.warning_time_before_poc)) / 5)
        bins = list(np.arange(-2, max_bin, 0.5))
        ax1.set_title(label="Time Warning-Message is received", fontsize=20)

        ax1.hist(self.warning_time_before_poc, bins=bins, color="orange", alpha=0.2)
        # ax1.set_xlim([-2, 5])
        ax1.axvline(x=0, linestyle="-", color="black")
        ax1.axvline(x=avg_cwa_time_before_warning, linestyle="--", linewidth=3, color="black")
        if len(self.warning_time_before_poc) > 10:
            ax1_twinx = ax1.twinx()
            density = gaussian_kde(self.warning_time_before_poc)
            xs = np.linspace(-2, 5, 200)
            density.covariance_factor = lambda: .25
            density._compute_covariance()
            ax1_twinx.plot(xs, density(xs), color="darkorange", linewidth=2)

        ax2.set_title(label="Time Collision-Message is received", fontsize=20)
        ax2.hist(self.collision_time_before_poc, bins=bins, color="red", alpha=0.2)
        # ax2.set_xlim([-2, 5])
        ax2.axvline(x=0, linestyle="-", color="black")
        ax2.axvline(x=avg_cwa_time_before_collision, linestyle="--", linewidth=3, color="black")
        if len(self.collision_time_before_poc) > 10:
            ax2_twinx = ax2.twinx()
            density = gaussian_kde(self.collision_time_before_poc)
            xs = np.linspace(-2, 5, 200)
            density.covariance_factor = lambda: .25
            density._compute_covariance()
            ax2_twinx.plot(xs, density(xs), color="darkred", linewidth=2)

        plt.tight_layout()
        plt.show()

        # --- plot matrix ---
        real_labels = ["Real:NoRisk", "Real:Warning", "Real:Collision"]
        cwa_labels = ["CWA:NoRisk", "CWA:Warning", "CWA:Collision"]

        fig, ax3 = plt.subplots(figsize=(7, 7))
        confusion_matrix = np.zeros(shape=(3, 3), dtype=int)

        for (real_value, pred_value) in self.real_pred_value:
            i, j = None, None

            if real_value == Risk.NoRisk: i = 0
            if real_value == Risk.Warning: i = 1
            if real_value == Risk.Collision: i = 2

            if pred_value == Risk.NoRisk: j = 0
            if pred_value == Risk.Warning: j = 1
            if pred_value == Risk.Collision: j = 2

            print("i:", i, " j:", j)
            print(confusion_matrix[j][i])
            if not (i is None and j in None):
                confusion_matrix[j][i] += 1
            print(confusion_matrix[j][i])

        print("----------------------")
        print("Confusion Error Matrix")
        print(confusion_matrix)
        print("----------------------")

        ax3.imshow(confusion_matrix, cmap='Greens')
        # Show all ticks and label them with the respective list entries
        ax3.set_xticks(np.arange(len(real_labels)), labels=real_labels)
        ax3.set_yticks(np.arange(len(cwa_labels)), labels=cwa_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax3.get_xticklabels(), ha="center", va="center")

        # Loop over data dimensions and create text annotations.
        for i in range(len(cwa_labels)):
            for j in range(len(real_labels)):
                text = ax3.text(j, i, round(confusion_matrix[i, j]),
                                ha="center", va="center", size=15, weight='bold', color="black")

        ax3.set_title("Confusion-Matrix for CWA Warnings")

        plt.tight_layout()
        plt.show()

    def evaluate_bike_car_pair(self, bike, car):
        distance_on_poc, on_poc_time, time_diff_on_poc = self.get_distance_on_poc(bike, car)
        if time_diff_on_poc == 0:
            highest_warning_real = Risk.Collision
        elif time_diff_on_poc < 2.5:
            highest_warning_real = Risk.Warning
        else:
            highest_warning_real = Risk.NoRisk

        times, risk_assessments = zip(*bike.cwa.risk_assessment_history)

        warning_set = False
        collision_set = False
        time_warning_cwa = None
        time_collision_cwa = None

        for time, risk_assessment in zip(times, risk_assessments):
            if car.vehicle_id in risk_assessment.keys():
                if risk_assessment[car.vehicle_id] == Risk.Warning and not warning_set:
                    time_warning_cwa = time
                    warning_set = True
                if risk_assessment[car.vehicle_id] == Risk.Collision and not collision_set:
                    time_collision_cwa = time
                    collision_set = True

        highest_warning_cwa = Risk.NoRisk
        if time_warning_cwa:
            highest_warning_cwa = Risk.Warning
            if not highest_warning_real == Risk.NoRisk:
                self.warning_time_before_poc.append(on_poc_time - time_warning_cwa)
        if time_collision_cwa:
            highest_warning_cwa = Risk.Collision
            if not highest_warning_real == Risk.NoRisk:
                self.collision_time_before_poc.append(on_poc_time - time_collision_cwa)

        self.real_pred_value.append([highest_warning_real, highest_warning_cwa])

        return highest_warning_real, highest_warning_cwa

    def get_real_distances(self, bike, car):
        # real path
        y_bike_real_path, x_bike_real_path, time_bike = zip(*bike.real_path)
        y_car_real_path, x_car_real_path, time_car = zip(*car.real_path)

        distances_real = []
        distances_real_time = []
        for x_bike, y_bike, t_bike in zip(x_bike_real_path, y_bike_real_path, time_bike):
            if t_bike >= time_car[0]:
                last_time = [x for x in time_car if x <= t_bike][-1]
                last_i = time_car.index(last_time)
                x_car = x_car_real_path[last_i]
                y_car = y_car_real_path[last_i]
                distances_real.append(math.dist([x_bike, y_bike], [x_car, y_car]))
                distances_real_time.append(t_bike)

        return distances_real_time, distances_real

    def get_poc(self, bike, car):
        # real path
        y_bike_real_path, x_bike_real_path, time_bike = zip(*bike.real_path)
        y_car_real_path, x_car_real_path, time_car = zip(*car.real_path)

        if len(time_bike) < 600:
            # collision
            min_dist = 0
            min_dist_time = time_bike[-1]
            time_diff_on_poc = abs(time_bike[-1] - time_car[-1])
            return min_dist, min_dist_time, time_diff_on_poc

        time_bike = [round(t, 2) for t in time_bike]
        time_car = [round(t, 2) for t in time_car]

        pos_bike = list(zip(x_bike_real_path, y_bike_real_path))
        pos_car = list(zip(x_car_real_path, y_car_real_path))

        nearest_points = []
        min_dist = 999
        for i_b, p_b in enumerate(pos_bike):
            for i_c, p_c in enumerate(pos_car):
                dist = math.dist(p_b, p_c)
                if dist < min_dist:
                    min_dist = dist
                    nearest_points = [p_b, p_c]

        poc = ((nearest_points[0][0] + nearest_points[1][0]) / 2, (nearest_points[0][1] + nearest_points[1][1]) / 2)
        return poc

    def get_distance_on_poc(self, bike, car):
        # real path
        y_bike_real_path, x_bike_real_path, time_bike = zip(*bike.real_path)
        y_car_real_path, x_car_real_path, time_car = zip(*car.real_path)

        if len(time_bike) < 600:
            # collision
            min_dist = 0
            min_dist_time = time_bike[-1]
            time_diff_on_poc = abs(time_bike[-1] - time_car[-1])
            return min_dist, min_dist_time, time_diff_on_poc

        time_bike = [round(t, 2) for t in time_bike]
        time_car = [round(t, 2) for t in time_car]

        pos_bike = list(zip(x_bike_real_path, y_bike_real_path))
        pos_car = list(zip(x_car_real_path, y_car_real_path))

        nearest_points = []
        min_dist = 999
        for i_b, p_b in enumerate(pos_bike):
            for i_c, p_c in enumerate(pos_car):
                dist = math.dist(p_b, p_c)
                if dist < min_dist:
                    min_dist = dist
                    nearest_points = [p_b, p_c]
                    nearest_times = [time_bike[i_b], time_car[i_c]]

        poc = ((nearest_points[0][0] + nearest_points[1][0]) / 2, (nearest_points[0][1] + nearest_points[1][1]) / 2)

        i_time = time_bike.index(nearest_times[1])
        corresponding_bike_pos = (x_bike_real_path[i_time], y_bike_real_path[i_time])

        i_time = time_car.index(nearest_times[0])
        corresponding_car_pos = (x_car_real_path[i_time], y_car_real_path[i_time])

        if nearest_times[0] < nearest_times[1]:
            min_dist = math.dist(poc, corresponding_car_pos)
            min_dist_time = nearest_times[0]
        else:
            min_dist = math.dist(poc, corresponding_bike_pos)
            min_dist_time = nearest_times[1]
        time_diff_on_poc = abs(nearest_times[0] - nearest_times[1])
        return min_dist, min_dist_time, time_diff_on_poc

    def calculate_segment_intersection(self, segment_1, segment_2):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        # Return true if line segments AB and CD intersect
        def intersect(a, b, c, d):
            return not (ccw(a, c, d) == ccw(b, c, d)) and (not (ccw(a, b, c) == ccw(a, b, d)))

        if not intersect(segment_1[0], segment_1[1], segment_2[0], segment_2[1]):
            return None

        xdiff = (segment_1[0][0] - segment_1[1][0], segment_2[0][0] - segment_2[1][0])
        ydiff = (segment_1[0][1] - segment_1[1][1], segment_2[0][1] - segment_2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        dd = (det(*segment_1), det(*segment_2))
        x = det(dd, xdiff) / div
        y = det(dd, ydiff) / div
        return x, y

    def plot_vehicle_paths(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        AXIS_LIM = [475, 542]

        for vehicle_id in self.vehicles.keys():
            real_path = self.vehicles[vehicle_id].real_path
            y_real_path, x_real_path, _ = zip(*real_path)

            if self.vehicles[vehicle_id].get_type() == "bike":
                color = "green"
                vehicle_type = "bike"
            elif self.vehicles[vehicle_id].get_type() == "car":
                color = "red"
                vehicle_type = "car"

            plt.plot(x_real_path, y_real_path, color=color, linestyle="-", label=str("Real Path " + vehicle_type))

            cam_path = self.vehicles[vehicle_id].cam_path
            y_cam_path, x_cam_path, time_cam_path = zip(*cam_path)
            ax.plot(x_cam_path, y_cam_path, color=color, linestyle="--",
                    label=str("Positions from CAM " + vehicle_type))
            ax.scatter(x_cam_path, y_cam_path, color=color, s=160, alpha=0.2, label=str("CAMs " + vehicle_type))

            gps_path = self.vehicles[vehicle_id].gps_path
            y_gps_path, x_gps_path, time_gps_path = zip(*gps_path)
            ax.scatter(x_gps_path, y_gps_path, color=str("dark" + color), s=50, label=str("GPS fixes" + vehicle_type))

            ax.set_xlim(AXIS_LIM)
            ax.set_ylim(AXIS_LIM)

            major_ticks = np.arange(AXIS_LIM[0], AXIS_LIM[1], 10)
            minor_ticks = np.arange(AXIS_LIM[0], AXIS_LIM[1], 2)

            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)

            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_distance_between_vehicles(self, bike=None, car=None, only_cwa=False):

        fig, ax = plt.subplots(figsize=(12, 6))

        if bike is None or car is None:

            for vehicle_id in self.vehicles.keys():

                if self.vehicles[vehicle_id].get_type() == "bike":
                    bike = self.vehicles[vehicle_id]
                elif self.vehicles[vehicle_id].get_type() == "car":
                    car = self.vehicles[vehicle_id]

            if bike is None or car is None:
                return None

        distances_real_time, distances_real = self.get_real_distances(bike, car)
        first_time = distances_real_time[0]
        distances_real_time = [t - first_time for t in distances_real_time]

        ax.plot(distances_real_time, distances_real, color="black", linestyle="-", linewidth="4", alpha=0.4,
                label="Distance calculated from real Positions")


        if not only_cwa:
            # GPS path
            y_bike_gps_path, x_bike_gps_path, time_gps_bike = zip(*bike.gps_path)
            y_car_gps_path, x_car_gps_path, time_gps_car = zip(*car.gps_path)

            distances_gps = []
            distances_gps_time = []
            for x_bike, y_bike, t_bike in zip(x_bike_gps_path, y_bike_gps_path, time_gps_bike):
                if t_bike >= time_gps_car[0]:
                    last_time = [x for x in time_gps_car if x <= t_bike][-1]
                    last_i = time_gps_car.index(last_time)
                    x_car = x_car_gps_path[last_i]
                    y_car = y_car_gps_path[last_i]
                    distances_gps.append(math.dist([x_bike, y_bike], [x_car, y_car]))
                    distances_gps_time.append(t_bike - first_time)

            ax.plot(distances_gps_time, distances_gps, color="blue", linestyle="-", linewidth="2",
                    label="Distance calculated from GPS Positions")

            # cam path
            received_car_cams = bike.received_cams[car.vehicle_id]
            received_car_cams_times = [cam.creation_time for cam in received_car_cams]

            distances_cam = []
            distances_cam_time = []
            for x_bike, y_bike, t_bike in zip(x_bike_gps_path, y_bike_gps_path, time_gps_bike):
                if t_bike >= received_car_cams_times[0]:
                    last_time = [x for x in received_car_cams_times if x <= t_bike][-1]
                    last_i = received_car_cams_times.index(last_time)
                    x_c = received_car_cams[last_i].latitude
                    y_c = received_car_cams[last_i].longitude
                    distances_cam.append(math.dist([x_bike, y_bike], [x_c, y_c]))
                    distances_cam_time.append(t_bike - first_time)

            ax.plot(distances_cam_time, distances_cam, color="orange", linestyle="-",
                    label="Distance calculated from CAMs")

        # interpolated path
        if only_cwa:
            interpolated_pos_bike = bike.cwa.danger_zones[car.vehicle_id].current_pos_1
            interpolated_pos_car = bike.cwa.danger_zones[car.vehicle_id].current_pos_2
            interpolated_time = bike.cwa.danger_zones[car.vehicle_id].current_pos_time

            distances_interpolated = []
            distances_interpolated_time = []
            for pos_bike, t_bike in zip(interpolated_pos_bike, interpolated_time):
                if t_bike >= interpolated_time[0]:
                    last_time = [x for x in interpolated_time if x <= t_bike][-1]
                    last_i = interpolated_time.index(last_time)
                    pos_car = interpolated_pos_car[last_i]
                    distances_interpolated.append(math.dist(pos_bike, pos_car))
                    distances_interpolated_time.append(t_bike - first_time)

            ax.plot(distances_interpolated_time, distances_interpolated, color="green", linestyle="-",
                    label="Distance calculated from cwa-interpolated position")


        # warning status
        warning_status_bike = []
        times, risk_assessments = zip(*bike.cwa.risk_assessment_history)
        risk_assessment_values = [d.values() for d in risk_assessments]
        for t, l in zip(times, risk_assessment_values):
            if len(l) > 0:
                max_risk = Risk(max([v.value for v in l]))
            else:
                max_risk = Risk.NoRisk
            warning_status_bike.append([t, max_risk])

        current_ws = warning_status_bike[0][1]
        start_ws_time = warning_status_bike[0][0]

        for ws in warning_status_bike:
            if ws[1] == current_ws:
                continue
            else:
                if current_ws.value == 0:
                    pass
                elif current_ws.value == 1:
                    ax.axvspan(start_ws_time, ws[0], alpha=0.3, color='yellow')
                elif current_ws.value == 2:
                    ax.axvspan(start_ws_time, ws[0], alpha=0.3, color='red')

                current_ws = ws[1]
                start_ws_time = ws[0]

        if not only_cwa:
            XLIM = [round(min(min(distances_real_time), min(distances_cam_time)) / 10) * 10, 75]
            ax.set_xlim(XLIM)
        else:
            XLIM = [round(min(min(distances_real_time), min(distances_interpolated)) / 10) * 10, 25]
            ax.set_xlim(XLIM)
        YLIM = [0, 150]

        ax.set_ylim(YLIM)

        major_x_ticks = np.arange(XLIM[0], XLIM[1], 5)
        minor_ticks = np.arange(XLIM[0], XLIM[1], 1)
        minor_y_ticks = np.arange(XLIM[0], XLIM[1], 5)

        ax.set_xticks(major_x_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        # ax.set_yticks(major_y_ticks)
        ax.set_yticks(minor_y_ticks, minor=True)

        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.3)

        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def plot_cwa_parameter(self):
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id].get_type() == "bike":
                bike = self.vehicles[vehicle_id]
            elif self.vehicles[vehicle_id].get_type() == "car":
                car = self.vehicles[vehicle_id]

        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 2, 2, 3]})
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]
        ax4 = ax[3]

        ax1.scatter(bike.cwa.danger_zones[car.vehicle_id].update_times,
                    bike.cwa.danger_zones[car.vehicle_id].danger_value)
        ax1.set_ylim([0, 200])
        ax1.set_title("danger_value")

        ax2.scatter(bike.cwa.danger_zones[car.vehicle_id].update_times,
                    bike.cwa.danger_zones[car.vehicle_id].diff_in_ttc)
        ax2.set_ylim([0, 8])
        ax2.set_title("diff_in_tcc")

        ax3.scatter(bike.cwa.danger_zones[car.vehicle_id].update_times,
                    bike.cwa.danger_zones[car.vehicle_id].danger_zone_size,
                    color="black", alpha=0.2)
        ax3.set_title("danger_zone_size")

        # warning status
        warning_status_bike = []
        times, risk_assessments = zip(*bike.cwa.risk_assessment_history)
        risk_assessment_values = [d.values() for d in risk_assessments]
        for t, l in zip(times, risk_assessment_values):
            if len(l) > 0:
                max_risk = Risk(max([v.value for v in l]))
            else:
                max_risk = Risk.NoRisk
            warning_status_bike.append([t, max_risk])

        current_ws = warning_status_bike[0][1]
        start_ws_time = warning_status_bike[0][0]

        for i, ws in enumerate(warning_status_bike):
            if ws[1] == current_ws and (not i == len(warning_status_bike) - 1):
                continue
            else:
                if current_ws.value == 0:
                    pass
                elif current_ws.value == 1:
                    ax4.axvspan(start_ws_time, ws[0], alpha=0.3, color='yellow')
                elif current_ws.value == 2:
                    ax4.axvspan(start_ws_time, ws[0], alpha=0.3, color='red')

                current_ws = ws[1]
                start_ws_time = ws[0]

        ax4.scatter(bike.cwa.danger_zones[car.vehicle_id].update_times,
                    bike.cwa.danger_zones[car.vehicle_id].prob_collision)
        ax4_2 = ax4.twinx()
        real_distances_time, real_distances = self.get_real_distances(bike, car)
        ax4_2.plot(real_distances_time, real_distances, color="black", alpha=0.4)
        ax4_2.plot(real_distances_time[real_distances.index(min(real_distances))], min(real_distances),
                   marker="o", markersize=10, color="black", alpha=0.6)
        plt.text(real_distances_time[real_distances.index(min(real_distances))], min(real_distances),
                 str("  ") + str(round(min(real_distances), 2)))
        ax4.set_title("prob_collisions")
        ax4.set_ylim([0, 1])

        plt.setp(ax, xlim=[bike.cwa.danger_zones[car.vehicle_id].update_times[0],
                           bike.cwa.danger_zones[car.vehicle_id].update_times[-1] + 5])

        plt.tight_layout()
        plt.show()

    def plot_cwa_prob(self, car_bike_pairs, title):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [1]})
        ax.set_title(title)

        for _, (car_id, bike_id) in car_bike_pairs.items():
            car = self.vehicles[car_id]
            bike = self.vehicles[bike_id]

            min_dist, min_dist_time, time_diff_on_poc = self.get_distance_on_poc(bike, car)
            print("min_dist", min_dist)

            if not car.vehicle_id in bike.cwa.danger_zones.keys():
                continue

            update_times = bike.cwa.danger_zones[car.vehicle_id].update_times
            prob_collision = bike.cwa.danger_zones[car.vehicle_id].prob_collision
            time_prob = zip(update_times, prob_collision)
            time_prob = [(t - min_dist_time, p) for t, p in time_prob if min_dist_time - 10 < t < min_dist_time]

            ax.plot(*zip(*time_prob))

        plt.setp(ax, ylim=[0, 1])

        plt.tight_layout()
        plt.show()

    def analyse_cwa_parameter(self, car_bike_pairs=None, title=""):
        if car_bike_pairs is None:
            car_bike_pairs = {}

            for vehicle_id in self.vehicles.keys():
                v_id = vehicle_id.split("_")[-1]
                if v_id in car_bike_pairs.keys():
                    if "bike" in vehicle_id:
                        car_bike_pairs[v_id].append(vehicle_id)
                    else:
                        car_bike_pairs[v_id].insert(0, vehicle_id)
                else:
                    car_bike_pairs[v_id] = [vehicle_id]

        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 20), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        fig.suptitle(title)
        ax1 = ax[0]
        ax1.set_title("collision_prob for Situations with time_diff_on_poc > 4 sec")
        ax2 = ax[1]
        ax2.set_title("collision_prob for Situations with time_diff_on_poc < 4 sec")
        ax3 = ax[2]
        ax3.set_title("collision_prob for Situations with time_diff_on_poc < 2 sec")
        ax4 = ax[3]
        ax4.set_title("collision_prob for Situations with time_diff_on_poc == 0 sec")

        for _, (car_id, bike_id) in car_bike_pairs.items():
            car = self.vehicles[car_id]
            bike = self.vehicles[bike_id]

            min_dist, min_dist_time, time_diff_on_poc = self.get_distance_on_poc(bike, car)
            print("min_dist", min_dist)

            if 0 in bike.cwa.movement_own["speed"]:
                continue

            if time_diff_on_poc == 0:
                ax_plot = ax4
            elif time_diff_on_poc < 2:
                ax_plot = ax3
            elif time_diff_on_poc < 4:
                ax_plot = ax2
            else:
                ax_plot = ax1

            if not car.vehicle_id in bike.cwa.danger_zones.keys():
                continue

            update_times = bike.cwa.danger_zones[car.vehicle_id].update_times
            prob_collision = bike.cwa.danger_zones[car.vehicle_id].prob_collision
            time_prob = zip(update_times, prob_collision)
            time_prob = [(t - min_dist_time, p) for t, p in time_prob if min_dist_time - 10 < t < min_dist_time]

            ax_plot.plot(*zip(*time_prob))

        plt.setp(ax, ylim=[0, 1])

        plt.tight_layout()
        plt.show()

    def plot_poc_error(self):
        for vehicle_id in self.vehicles.keys():
            if self.vehicles[vehicle_id].get_type() == "bike":
                bike = self.vehicles[vehicle_id]
            elif self.vehicles[vehicle_id].get_type() == "car":
                car = self.vehicles[vehicle_id]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax1 = ax
        ax1.set_title("Divergence of POC Prediction over time")

        real_poc = self.get_poc(bike, car)
        print(real_poc)
        poc_predictions = bike.cwa.danger_zones[car.vehicle_id].poc
        print(poc_predictions)

        poc_deviation = []
        for i, p in enumerate(poc_predictions):
            d = (p[0] - real_poc[0], p[1] - real_poc[1])
            poc_deviation.append(d)

        ax1.scatter(x=list(zip(*poc_deviation))[0], y=list(zip(*poc_deviation))[1],
                    c=np.arange(len(poc_deviation)), cmap="viridis")

        plt.hlines(y=0, xmin=-3, xmax=3, colors="black", alpha=0.2)
        plt.vlines(x=0, ymin=-3, ymax=3, colors="black", alpha=0.2)

        plt.tight_layout()
        plt.show()
