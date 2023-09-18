import os
from pathlib import Path

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 4), dpi=200,
                               gridspec_kw={'height_ratios': [2, 2, 4]}, sharex=True)
        # fig.subplots_adjust(hspace=0)

        # x_lim = [0, 15]

        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]
        ax3_2 = ax3.twinx()

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

        ax3_2.plot(time, distance_vehicles, color="black", alpha=0.4)
        ax3_2.plot(time[distance_vehicles[x_lim[0]*10:x_lim[1]*10].index(min(distance_vehicles[x_lim[0]*10:x_lim[1]*10]))],
                   min(distance_vehicles[x_lim[0]*10:x_lim[1]*10]),
                   marker="o", markersize=10, color="black", alpha=0.6)
        plt.text(time[distance_vehicles[x_lim[0]*10:x_lim[1]*10].index(min(distance_vehicles[x_lim[0]*10:x_lim[1]*10]))],
                 min(distance_vehicles[:140]),
                 str("  ") + str(round(min(distance_vehicles) - 1.94, 2)))

        ax1.set_ylabel("Summe der \n DangerValue")
        ax2.set_ylabel("Differenz der \n TTC in Sek.")
        ax2.set_ylim([0, 10])
        ax3.set_ylabel("Kollisions- \n wahrscheinlichkeit")
        ax3.set_xlabel("Zeit in Sekunden")
        ax3_2.set_ylabel("Abstand in Meter")

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

        ax3.imshow(confusion_matrix, cmap='Greens')
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
        time = list(self.cwa_df.loc[:, "current_time"])
        # convert to seconds with start at 0
        time = [(t - time[0]) / 1000000 for t in time]

        delta_time = [time[i] - time[i-1] for i in range(1, len(time))]
        plt.plot(time[1:], delta_time)

        df_warning = self.cwa_df.loc[(self.cwa_df["risk"] == "Warning")]
        time_warning = []
        for i, row in df_warning.iterrows():
            time_warning.append(row["distance_to_poc_own"] / row["current_speed_own"])
        print("time_warning: ", time_warning)

        time = list(self.cwa_df.loc[:, "current_time"])
        # convert to seconds with start at 0
        time = [(t - time[0]) / 1000000 for t in time]

        danger_value = []
        speed_own = list(self.cwa_df.loc[:, "current_speed_own"])
        distance_to_poc_own = list(self.cwa_df.loc[:, "distance_to_poc_own"])
        for i in range(len(distance_to_poc_own)):
            dv = calculate_danger_value(dist_to_poc=distance_to_poc_own[i], dangerZone_width=1, speed=speed_own[i],
                                        normal_brake_distance=6, emergency_brake_distance=3)
            danger_value.append(dv * 2.5)

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

        fig = plt.figure(figsize=(8, 2), dpi=200)
        ax = fig.add_subplot(111)
        ax_2 = ax.twinx()

        ax.set_ylim([0, 1.1])
        ax.fill_between(x=time, y1=0.4, y2=0.7, color="yellow", alpha=0.2)
        ax.fill_between(x=time, y1=0.7, y2=1.1, color="red", alpha=0.2)
        ax.scatter(time, prob_collision, color="darkred")

        ax_2.plot(time, distance_vehicles, color="grey")

        plt.tight_layout()
        plt.show()