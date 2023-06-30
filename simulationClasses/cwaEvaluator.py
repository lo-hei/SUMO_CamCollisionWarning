import math

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
        self.evaluation_mode = evaluation_mode
        self.runs = runs

        self.route_manager = RouteManager()
        if evaluation_mode in [0, 1, 2, 3]:
            self.generate_routefile()

        if evaluation_mode == 4:
            self.generate_evaluation_routefile()

        # just inactive vehicles from simulationManager
        self.vehicles = None

        self.real_pred_value = []  # [[real value, predicted value], [r_v, p_v], ... ]
        self.cwa_time_before_warning = []
        self.cwa_time_before_collision = []

    def generate_routefile(self):
        get_bike_left_car_straight = self.route_manager.get_bike_straight_car_straight(repeats=self.runs)

    def generate_evaluation_routefile(self):
        evaluation_scenario = self.route_manager.get_evaluation_scenario(repeats=self.runs)

    def evaluate(self):
        if self.evaluation_mode == 0:
            return None
        elif self.evaluation_mode == 1:
            self.plot_vehicle_paths()
        elif self.evaluation_mode == 2:
            self.plot_distance_between_vehicles()
        elif self.evaluation_mode == 3:
            self.plot_distance_bike_view()
        elif self.evaluation_mode == 4:
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

        for _, (car_id, bike_id) in car_bike_pairs.items():
            car = self.vehicles[car_id]
            bike = self.vehicles[bike_id]

            self.evaluate_bike_car_pair(bike, car)

        if len(self.cwa_time_before_warning) > 0:
            avg_cwa_time_before_warning = sum(self.cwa_time_before_warning) / len(self.cwa_time_before_warning)
        else:
            avg_cwa_time_before_warning = 0
        if len(self.cwa_time_before_collision) > 0:
            avg_cwa_time_before_collision = sum(self.cwa_time_before_collision) / len(self.cwa_time_before_collision)
        else:
            avg_cwa_time_before_collision = 0

        # --- plot histograms ---
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 1]})
        ax1 = ax[0]
        ax2 = ax[1]

        max_bin = 5 * round(max(max(self.cwa_time_before_collision), max(self.cwa_time_before_warning))/5)
        bins = list(np.arange(-2, max_bin, 0.5))
        ax1.set_title(label="Time Warning-Message is received", fontsize=20)

        ax1.hist(self.cwa_time_before_warning, bins=bins, color="orange")
        ax1.axvline(x=0, linestyle="-", color="black")
        ax1.axvline(x=avg_cwa_time_before_warning, linestyle="--", linewidth=5, color="black")
        if len(self.cwa_time_before_warning) > 10:
            ax1_twinx = ax1.twinx()
            density = gaussian_kde(self.cwa_time_before_warning)
            xs = np.linspace(-2, 10, 200)
            density.covariance_factor = lambda: .25
            density._compute_covariance()
            ax1_twinx.plot(xs, density(xs), color="darkorange", linewidth=2)

        ax2.set_title(label="Time Collision-Message is received", fontsize=20)
        ax2.hist(self.cwa_time_before_collision, bins=bins, color="red")
        ax2.axvline(x=0, linestyle="-", color="black")
        ax2.axvline(x=avg_cwa_time_before_collision, linestyle="--", linewidth=5, color="black")
        if len(self.cwa_time_before_collision) > 10:
            ax2_twinx = ax2.twinx()
            density = gaussian_kde(self.cwa_time_before_collision)
            xs = np.linspace(-2, 20, 200)
            density.covariance_factor = lambda: .25
            density._compute_covariance()
            ax2_twinx.plot(xs, density(xs), color="darkred", linewidth=2)

        plt.tight_layout()
        plt.show()

        print("self.cwa_time_before_warning:", self.cwa_time_before_warning)
        print("self.cwa_time_before_collision:", self.cwa_time_before_collision)

        # --- plot matrix ---
        real_labels = ["Real:NoRisk", "Real:Warning", "Real:Collision"]
        cwa_labels = ["CWA:NoRisk", "CWA:Warning", "CWA:Collision"]

        fig, ax3 = plt.subplots(figsize=(7, 7))
        confusion_matrix = np.ndarray(shape=(3, 3))

        for (real_value, pred_value) in self.real_pred_value:
            if real_value == Risk.NoRisk: i = 0
            if real_value == Risk.Warning: i = 1
            if real_value == Risk.Collision: i = 2

            if pred_value == Risk.NoRisk: j = 0
            if pred_value == Risk.Warning: j = 1
            if pred_value == Risk.Collision: j = 2

            confusion_matrix[j, i] = round(confusion_matrix[j, i] + 1)

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
        time_collision_real, time_warning_real = self.get_collision_and_warning_times(bike.real_path, car.real_path)
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

        highest_warning_real = Risk.NoRisk
        highest_warning_cwa = Risk.NoRisk
        if time_warning_real:
            highest_warning_real = Risk.Warning
        if time_collision_real:
            highest_warning_real = Risk.Collision
        if time_warning_cwa:
            highest_warning_cwa = Risk.Warning
        if time_collision_cwa:
            highest_warning_cwa = Risk.Collision

        self.real_pred_value.append([highest_warning_real, highest_warning_cwa])

        if time_warning_cwa and time_warning_real:
            if (time_warning_real - time_warning_cwa) > 20:
                self.plot_cwa_cam_distance(bike=bike, car=car)
            else:
                self.cwa_time_before_warning.append(time_warning_real - time_warning_cwa)
        if time_collision_cwa and time_collision_real:
            if (time_collision_real - time_collision_cwa) > 20:
                self.plot_cwa_cam_distance(bike=bike, car=car)
            else:
                self.cwa_time_before_collision.append(time_collision_real - time_collision_cwa)

    def get_collision_and_warning_times(self, bike_path, car_path):
        distances_real = []
        distances_real_time = []

        lat_car_real_path, lon_car_real_path, time_car = zip(*car_path)

        for lat_bike, lon_bike, time_bike in bike_path:
            if time_bike >= time_car[0]:
                nearest_car_time = [t for t in time_car if t <= time_bike][-1]
                nearest_car_i = time_car.index(nearest_car_time)
                lat_car = lat_car_real_path[nearest_car_i]
                lon_car = lon_car_real_path[nearest_car_i]

                distances_real.append(math.dist([lat_bike, lon_bike], [lat_car, lon_car]))
                distances_real_time.append(time_bike)

        i_real_collision = [i for i, d in enumerate(distances_real) if d < self.distance_collision]
        i_real_warning = [i for i, d in enumerate(distances_real) if d < self.distance_warning]

        if i_real_collision:
            time_collision = distances_real_time[i_real_collision[0]]
        else:
            time_collision = None

        if i_real_warning:
            time_warning = distances_real_time[i_real_warning[0]]
        else:
            time_warning = None

        return time_collision, time_warning

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
            ax.plot(x_cam_path, y_cam_path, color=color, linestyle="--", label=str("Positions from CAM " + vehicle_type))
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

    def plot_distance_between_vehicles(self, bike=None, car=None):

        fig, ax = plt.subplots(figsize=(12, 6))

        if bike is None or car is None:

            for vehicle_id in self.vehicles.keys():

                if self.vehicles[vehicle_id].get_type() == "bike":
                    bike = self.vehicles[vehicle_id]
                elif self.vehicles[vehicle_id].get_type() == "car":
                    car = self.vehicles[vehicle_id]

            if bike is None or car is None:
                return None

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

        ax.plot(distances_real_time, distances_real, color="black", linestyle="-", linewidth="4", alpha=0.4,
                label="Distance calculated from real Positions")

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
                distances_gps_time.append(t_bike)

        ax.plot(distances_gps_time, distances_gps, color="blue", linestyle="-", linewidth="2",
                label="Distance calculated from GPS Positions")

        # cam path
        received_car_cams = bike.received_cams[car.vehicle_id]
        received_car_cams_times = [cam.creation_time for cam in received_car_cams]

        distances_cam = []
        distances_cam_time = []
        for x_bike, y_bike, t_bike in zip(x_bike_real_path, y_bike_real_path, time_bike):
            if t_bike >= received_car_cams_times[0]:
                last_time = [x for x in received_car_cams_times if x <= t_bike][-1]
                last_i = received_car_cams_times.index(last_time)
                x_c = received_car_cams[last_i].latitude
                y_c = received_car_cams[last_i].longitude
                distances_cam.append(math.dist([x_bike, y_bike], [x_c, y_c]))
                distances_cam_time.append(t_bike)

        ax.plot(distances_cam_time, distances_cam, color="orange", linestyle="-",
                label="Distance calculated from CAMs")

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

        XLIM = [round(min(min(distances_real_time), min(distances_cam_time))/10)*10, 75]
        YLIM = [0, 150]
        ax.set_xlim(XLIM)
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

    def plot_distance_bike_view(self, bike=None):
        pass
