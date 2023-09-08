import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from tools.transmissionAnalyser.utils.helper import distance_earth
from tools.transmissionAnalyser.utils.transmissionAnalyser import TransmissionAnalyser


class TransmissionPlotter(TransmissionAnalyser):

    def __init__(self, transmission_test_folder, gps_cam_log_file, start_if_gps_signal, end_in_sync):
        super(TransmissionPlotter, self).__init__(transmission_test_folder, gps_cam_log_file, start_if_gps_signal, end_in_sync)

    def plot_transmissions_on_map(self):
        outgoing_file = [self.evk_1_camFile_outgoing, self.evk_2_camFile_outgoing]
        incoming_file = [self.evk_2_camFile_incoming, self.evk_1_camFile_incoming]
        direction_string = ["Transmission Direction 1 (outgoing) -> 2 (incoming)",
                            "Transmission Direction 2 (outgoing) -> 1 (incoming)"]

        fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})

        for outgoing, incoming, direction, ax in zip(outgoing_file, incoming_file, direction_string, ax_list):

            outgoing_pos = outgoing.get_positions()
            incoming_pos = incoming.get_positions()

            if len(outgoing_pos[0]) == 0 or len(incoming_pos[0]) == 0:
                print("No Data found. Maybe not all EVKs got an GPS-Signal")
                continue

            ax.set_title(direction)
            ax.scatter(outgoing_pos[1], outgoing_pos[0], color="orangered",
                        marker="o", s=60, label="nicht erfolgreiche Zustellung")
            ax.scatter(incoming_pos[1], incoming_pos[0], color="darkgreen",
                        marker="o", s=60, label="erfolgreiche Zustellung")
            receive_rate = round(len(incoming_pos[0]) / len(outgoing_pos[0]), 3)

            ax.plot([], [], ' ', label=str("Gesendete CAMs: " + str(len(outgoing_pos[0]))))
            ax.plot([], [], ' ', label=str("Erfolgreich in %: " + str(receive_rate)))

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_transmission_time_on_distance(self):

        outgoing_file = [self.evk_1_camFile_outgoing, self.evk_2_camFile_outgoing]
        incoming_file = [self.evk_2_camFile_incoming, self.evk_1_camFile_incoming]
        position_receiver = [self.evk_2_gpsFile.get_positions(), self.evk_1_gpsFile.get_positions()]
        direction_string = ["Transmission Direction 1 (outgoing) -> 2 (incoming)",
                            "Transmission Direction 2 (outgoing) -> 1 (incoming)"]

        fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1]})

        for outgoing, incoming, pos_receiver, direction, ax in zip(outgoing_file, incoming_file,
                                                                   position_receiver, direction_string, ax_list):

            outgoing_times = outgoing.get_times()
            outgoing_pos = outgoing.get_positions()
            outgoing_ids = outgoing.get_extended_ids()

            incoming_times = incoming.get_times()
            incoming_ids = incoming.get_extended_ids()

            transmission_log = []

            for message_id, lat, lon, time in zip(outgoing_ids, outgoing_pos[0], outgoing_pos[1], outgoing_times):
                if message_id in incoming_ids:
                    received = True
                    receive_time = incoming_times[incoming_ids.index(message_id)]

                    # find index for position of receiver with same time
                    i = min(range(len(pos_receiver["gps_time"])), key=lambda j: abs(pos_receiver["gps_time"][j] - time))

                    distance = distance_earth(lat, lon, pos_receiver["latitude"][i], pos_receiver["longitude"][i])
                    transmission_time = receive_time - time

                    transmission_log.append(transmission_time)
                    ax.plot(distance, transmission_time, marker="o", color="blue")

            avg_transmission_time = sum(transmission_log) / len(transmission_log)
            transmission_log = [t for t in transmission_log if 0 < t < (2 * avg_transmission_time)]
            avg_transmission_time = sum(transmission_log) / len(transmission_log)
            ax.axhline(avg_transmission_time, linestyle="--")

            ax.set_title(direction)
            ax.set_xlabel("distance")
            ax.set_ylabel("transmission-time")

        plt.tight_layout()
        plt.show()

    def plot_transmission_success_on_distance(self):

        outgoing_file = [self.evk_1_camFile_outgoing, self.evk_2_camFile_outgoing]
        incoming_file = [self.evk_2_camFile_incoming, self.evk_1_camFile_incoming]
        position_receiver = [self.evk_2_gpsFile.get_positions(), self.evk_1_gpsFile.get_positions()]
        direction_string = ["Transmission Direction 1 (outgoing) -> 2 (incoming)",
                            "Transmission Direction 2 (outgoing) -> 1 (incoming)"]

        fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1]})

        for outgoing, incoming, pos_receiver, direction, ax in zip(outgoing_file, incoming_file,
                                                                   position_receiver, direction_string, ax_list):

            outgoing_times = outgoing.get_times()
            outgoing_pos = outgoing.get_positions()
            outgoing_ids = outgoing.get_extended_ids()

            incoming_times = incoming.get_times()
            incoming_ids = incoming.get_extended_ids()

            # change time to passed time in seconds
            outgoing_times = [(t - outgoing_times[0]) / 1000000 for t in outgoing_times]

            for message_id, lat, lon, time in zip(outgoing_ids, outgoing_pos[0], outgoing_pos[1], outgoing_times):
                if message_id in incoming_ids:
                    received = True
                else:
                    received = False

                # find index for position of receiver with same time
                i = min(range(len(pos_receiver["gps_time"])), key=lambda j: abs(pos_receiver["gps_time"][j] - time))

                distance = distance_earth(lat, lon, pos_receiver["latitude"][i], pos_receiver["longitude"][i])

                if received:
                    ax.plot(time, distance, marker="o", color="darkgreen", markersize=3)
                else:
                    ax.plot(time, distance, marker="o", color="orangered", markersize=3)

            # ax.set_title(direction)
            ax.set_xlabel("Zeit in Sekunden")
            ax.set_ylabel("Distanz in Meter")

        plt.tight_layout()
        plt.show()

    def plot_transmission_accuracy_on_distance(self, bin_size):

        outgoing_file = [self.evk_1_camFile_outgoing, self.evk_2_camFile_outgoing]
        incoming_file = [self.evk_2_camFile_incoming, self.evk_1_camFile_incoming]
        position_receiver = [self.evk_2_gpsFile.get_positions(), self.evk_1_gpsFile.get_positions()]
        direction_string = ["Transmission Direction 1 (outgoing) -> 2 (incoming)",
                            "Transmission Direction 2 (outgoing) -> 1 (incoming)"]

        fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(18, 4), gridspec_kw={'width_ratios': [1, 1]})

        for outgoing, incoming, pos_receiver, direction, ax in zip(outgoing_file, incoming_file,
                                                                   position_receiver, direction_string, ax_list):

            outgoing_times = outgoing.get_times()
            outgoing_pos = outgoing.get_positions()
            outgoing_ids = outgoing.get_extended_ids()

            incoming_times = incoming.get_times()
            incoming_ids = incoming.get_extended_ids()

            transmission_log = []

            # change time to passed time in seconds
            outgoing_times = [(t - outgoing_times[0]) / 1000000 for t in outgoing_times]

            for message_id, lat, lon, time in zip(outgoing_ids, outgoing_pos[0], outgoing_pos[1], outgoing_times):
                if message_id in incoming_ids:
                    received = True
                else:
                    received = False

                # find index for position of receiver with same time
                i = min(range(len(pos_receiver["gps_time"])), key=lambda j: abs(pos_receiver["gps_time"][j] - time))

                distance = distance_earth(lat, lon, pos_receiver["latitude"][i], pos_receiver["longitude"][i])
                transmission_log.append([distance, received])

            transmission_log.sort(key=lambda x: x[0])
            transmission_accuracy = {}

            # bin-size in meters
            # last_bin = math.ceil(transmission_log[-1][0] / 50) * 50
            last_bin = 400
            bins = np.arange(0, last_bin, bin_size)

            num_transmissions_list = []
            xticks = []

            for bin in bins:
                num_transmissions = 0
                num_successful = 0
                xticks.append(str(bin))

                for log in transmission_log:
                    if bin <= log[0] < (bin + bin_size):
                        num_transmissions += 1
                        if log[1]:
                            num_successful += 1

                if num_transmissions > 0:
                    accuracy = num_successful / num_transmissions
                else:
                    accuracy = 0

                num_transmissions_list.append(num_transmissions)
                transmission_accuracy[bin] = accuracy

            print(transmission_accuracy)

            colors = cm.RdYlGn(np.array(list(transmission_accuracy.values())) / float(max(transmission_accuracy.values())))
            ax.bar(transmission_accuracy.keys(), transmission_accuracy.values(), width=bin_size*0.8, color=colors)

            for i in range(len(bins)):
                ax.text(bins[i], 0.05, num_transmissions_list[i], horizontalalignment='center',
                        verticalalignment='center', fontsize=10)

            ax.set_ylim([0, 1])
            # ax.set_title(direction)
            x_bins = [b - bin_size*0.5 for b in bins]
            ax.set_xticks(x_bins[::5])
            ax.set_xticklabels(xticks[::5])
            ax.set_xlabel("Distanz in Meter")
            ax.set_ylabel("Übertragungszuverlässigkeit")

        plt.tight_layout()
        plt.show()




