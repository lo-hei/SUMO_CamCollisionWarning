import os
from pathlib import Path

import geopy.distance
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def distance_earth(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    d = geopy.distance.geodesic(coords_1, coords_2).m
    return d


def calculate_danger_value(dist_to_poc, dangerZone_width, speed, normal_brake_distance, emergency_brake_distance):
    # if position is in DangerZone. return 100
    if dist_to_poc <= dangerZone_width:
        return 100

    time_of_normal_brake = normal_brake_distance / speed
    time_to_dist = dist_to_poc / speed

    if dist_to_poc < emergency_brake_distance:
        return 100

    dv_normal_brake = 50
    danger_range_decrease_by_1s = 3

    if dist_to_poc < normal_brake_distance:
        danger_value = dv_normal_brake + dv_normal_brake * (1 - (dist_to_poc - emergency_brake_distance) / (
                normal_brake_distance - emergency_brake_distance))
        return danger_value

    time_after_normal_break = time_to_dist - time_of_normal_brake
    danger_value = dv_normal_brake
    while danger_value > 0:
        if time_after_normal_break > 1:
            time_after_normal_break -= 1
            danger_value -= danger_range_decrease_by_1s
        elif 0 < time_after_normal_break < 1:
            danger_value -= danger_range_decrease_by_1s * time_after_normal_break
            return danger_value

    return 0


class CwaPlotter:

    def __init__(self, cwa_data_path):
        self.cwa_data_path = cwa_data_path
        self.cwa_df = None
        self.load_data()

    def load_data(self):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        root_path = current_path.parent.parent.__str__()
        file = "cwa_risk"
        file_path = root_path + "\\tools\\src\\cwaData" + "\\" + self.cwa_data_path + "\\" + file + ".csv"
        print(file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, skipinitialspace=True)
            print("--- loaded data ---")
            print(df)
            print("-------------------")
            self.cwa_df = df

    def plot_prob_collision(self):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 5), dpi=200,
                               gridspec_kw={'height_ratios': [2, 2, 4]}, sharex=True)
        # fig.subplots_adjust(hspace=0)

        x_lim = [0, 15]

        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]

        time = list(self.cwa_df.loc[:, "current_time"])
        # convert to seconds with start at 0
        time = [(t - time[0]) / 1000000 for t in time]

        danger_value = []
        speed_own = list(self.cwa_df.loc[:, "current_speed_own"])
        distance_to_poc_own = list(self.cwa_df.loc[:, "distance_to_poc_own"])
        for i in range(len(distance_to_poc_own)):
            print(i)
            dv = calculate_danger_value(dist_to_poc=distance_to_poc_own[i], dangerZone_width=1, speed=speed_own[i],
                                        normal_brake_distance=6, emergency_brake_distance=3)
            danger_value.append(dv*2)
        print("danger_value", danger_value)
        ax1.scatter(time, danger_value, c="darkblue")

        diff_ttc = list(self.cwa_df.loc[:, "diff_ttc"])
        ax2.scatter(time, diff_ttc, c="darkblue")
        prob_collision = list(self.cwa_df.loc[:, "prob_collision"])
        # correction by danger_value
        for i, p in enumerate(prob_collision):
            prob_collision[i] = prob_collision[i] * (danger_value[i] / 200)
        ax3.scatter(time, prob_collision, c="darkred")

        lat_own = list(self.cwa_df.loc[:, "current_lat_own"])
        lon_own = list(self.cwa_df.loc[:, "current_lon_own"])

        lat_other = list(self.cwa_df.loc[:, "current_lat_other"])
        lon_other = list(self.cwa_df.loc[:, "current_lon_other"])

        distance_vehicles = []

        for i in range(len(lat_other)):
            d = distance_earth(lat_own[i], lon_own[i], lat_other[i], lon_other[i])
            distance_vehicles.append(d)

        ax1.set_ylabel("Summe der \n DangerValue")
        ax2.set_ylabel("Differenz der \n TTC in Sek.")
        ax2.set_ylim([0, 10])
        ax3.set_ylabel("Kollisions- \n wahrscheinlichkeit")
        ax3.set_xlabel("Zeit in Sekunden")

        plt.tight_layout()
        if x_lim:
            plt.xlim(x_lim)
        plt.show()

    def plot_cwa_evaluation(self, confusion_matrix, warning_time_before_poc, collision_time_before_poc):
        # --- plot histograms ---
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 3), gridspec_kw={'height_ratios': [1, 1]})
        ax1 = ax[0]
        ax2 = ax[1]

        max_bin = 5 * round(max(max(collision_time_before_poc), max(warning_time_before_poc)) / 5)
        bins = list(np.arange(-2, max_bin, 0.5))

        avg_cwa_time_before_warning = sum(warning_time_before_poc) / len(warning_time_before_poc)
        avg_cwa_time_before_collision = sum(collision_time_before_poc) / len(collision_time_before_poc)

        ax1.hist(warning_time_before_poc, bins=bins, color="orange", alpha=0.2)
        ax1.set_xlim([-2, 5])
        ax1.set_ylabel("Häufigkeit")
        ax1.axvline(x=0, linestyle="-", color="black")
        ax1.axvline(x=avg_cwa_time_before_warning, linestyle="--", linewidth=3, color="black",
                    label=str(str(r'$\varnothing$') + " Vorwarnzeit (WARNUNG) in s: " + str(
                        round(avg_cwa_time_before_warning, 2))))
        if len(warning_time_before_poc) > 10:
            ax1_twinx = ax1.twinx()
            density = gaussian_kde(warning_time_before_poc)
            xs = np.linspace(-2, 5, 200)
            density.covariance_factor = lambda: .25
            density._compute_covariance()
            ax1_twinx.plot(xs, density(xs), color="darkorange", linewidth=2)

        ax2.hist(collision_time_before_poc, bins=bins, color="red", alpha=0.2)
        ax2.set_xlim([-2, 5])
        ax2.axvline(x=0, linestyle="-", color="black")
        ax2.set_xlabel("Zeit ausgehend von Kollision in s")
        ax2.set_ylabel("Häufigkeit")
        ax2.axvline(x=avg_cwa_time_before_collision, linestyle="--", linewidth=3, color="black",
                    label=str(str(r'$\varnothing$') + " Vorwarnzeit (KOLLISION) in s: " + str(
                        round(avg_cwa_time_before_collision, 2))))

        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        # --- plot matrix ---
        real_labels = ["Real:KeinRisiko", "Real:Warnung", "Real:Kollision"]
        cwa_labels = ["CWA:KeinRisiko", "CWA:Warnung", "CWA:Kollision"]

        fig, ax3 = plt.subplots(figsize=(4, 4))

        norm = plt.Normalize(-2, 2)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "#c8c8c8", "#ff6b00"])

        ax3.imshow(confusion_matrix, cmap=cmap)
        # Show all ticks and label them with the respective list entries
        ax3.set_xticks(np.arange(len(real_labels)), labels=real_labels)
        ax3.set_yticks(np.arange(len(cwa_labels)), labels=cwa_labels, rotation=90, va="center")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax3.get_xticklabels(), ha="center", va="center")

        # Loop over data dimensions and create text annotations.
        for i in range(len(cwa_labels)):
            for j in range(len(real_labels)):
                text = ax3.text(j, i, round(confusion_matrix[i, j]),
                                ha="center", va="center", size=15, weight='bold', color="black")

        plt.tight_layout()
        plt.show()

    def evaluate_tests(self):
        df_warning = self.cwa_df.loc[(self.cwa_df["risk"] == "Warning")]
        time_warning = []
        for i, row in df_warning.iterrows():
            time_warning.append(row["distance_to_poc_own"] / row["current_speed_own"])
        print("time_warning: ", time_warning)

        current_time = list(self.cwa_df.loc[:, "current_time"])
        # convert to seconds with start at 0
        time = [(t - current_time[0]) / 1000000 for t in current_time]

        # time = [(lambda i: t - 100 if t > 100 else t)(i)for t in time]

        danger_value = []
        speed_own = list(self.cwa_df.loc[:, "current_speed_own"])
        distance_to_poc_own = list(self.cwa_df.loc[:, "distance_to_poc_own"])
        for i in range(len(distance_to_poc_own)):
            dv = calculate_danger_value(dist_to_poc=distance_to_poc_own[i], dangerZone_width=1, speed=speed_own[i],
                                        normal_brake_distance=6, emergency_brake_distance=3)
            danger_value.append(dv * 2.2)

        diff_ttc = list(self.cwa_df.loc[:, "diff_ttc"])

        prob_collision = list(self.cwa_df.loc[:, "prob_collision"])
        # correction by danger_value
        for i, p in enumerate(prob_collision):
            prob_collision[i] = prob_collision[i] * (danger_value[i] / 200)

        lat_own = list(self.cwa_df.loc[:, "current_lat_own"])
        lon_own = list(self.cwa_df.loc[:, "current_lon_own"])

        lat_other = list(self.cwa_df.loc[:, "current_lat_other"])
        lon_other = list(self.cwa_df.loc[:, "current_lon_other"])

        distance_vehicles = []

        for i in range(len(lat_other)):
            d = distance_earth(lat_own[i], lon_own[i], lat_other[i], lon_other[i])
            distance_vehicles.append(d)

        # find intervalls for test-runs
        peaks, _ = find_peaks(distance_vehicles, distance=5)
        peaks = np.append(peaks, len(distance_vehicles) - 1)
        print("time_intervalls_test_runs", peaks)

        # filter for only highest Value in on test
        highest_prob_in_run = []
        highest_time_in_run = []
        time_to_poc_own = list(self.cwa_df.loc[:, "time_to_poc_own"])
        print("time_to_poc_own before:", time_to_poc_own)

        plt.plot(time_to_poc_own)
        plt.show()

        # correction of time_to_poc_own
        time_to_poc_own = [(current_time[i] + t_poc) / 1000000 for i, t_poc in enumerate(time_to_poc_own)]

        plt.plot(time_to_poc_own)
        plt.xlim([0, 50])
        plt.show()

        print("time_to_poc_own after:", time_to_poc_own)
        warning_time = []
        collision_time = []
        for i, p in enumerate(peaks):
            if i == 0:
                start = 0
            else:
                start = peaks[i - 1]

            time_to_poc_testrun = time_to_poc_own[start:p]
            for i_testrun, p_testrun in enumerate(prob_collision[start:p]):
                if p > 0.45:
                    warning_time.append(time_to_poc_testrun[i_testrun])
                    break
            for i_testrun, p_testrun in enumerate(prob_collision[start:p]):
                if p > 0.7:
                    collision_time.append(time_to_poc_testrun[i_testrun])
                    break

            highest_prob_in_run.append(max(prob_collision[start:p]))
            highest_time_in_run.append(time[start + prob_collision[start:p].index(highest_prob_in_run[-1])])

        print("warning_time:", warning_time)
        print("collision_time:", collision_time)

        fig = plt.figure(figsize=(8, 2.5), dpi=200)
        ax = fig.add_subplot(111)
        ax_2 = ax.twinx()

        ax.set_ylim([0, 1.1])
        ax_2.plot(time, distance_vehicles, color="grey", alpha=0.8)

        # plot intervalls
        for i, p in enumerate(peaks):
            if i in [7, 17]:
                ax.axvline(x=time[p], color='black', ls="--", alpha=0.8, linewidth=2)
            else:
                ax.axvline(x=time[p], color='black', ls="--", alpha=0.2, linewidth=1.5)

        ax.fill_between(x=time, y1=0.4, y2=0.7, color="#c8c8c8", alpha=0.4)
        ax.fill_between(x=time, y1=0.7, y2=1.1, color="#ff6b00", alpha=0.4)
        # ax.scatter(time, prob_collision, color="darkred")
        ax.scatter(highest_time_in_run, highest_prob_in_run, color="black")

        ax.set_ylabel("Kollisions- \n wahrscheinlichkeit")
        ax_2.set_ylabel("Abstand in Meter")
        ax.set_xlabel("Zeit in Sekunden")
        plt.tight_layout()
        plt.show()