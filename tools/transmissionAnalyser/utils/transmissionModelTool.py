import math

import numpy as np

from simulationClasses.TransmissionModel.transmissionModel import TransmissionModel
from tools.transmissionAnalyser.utils.helper import distance_earth
from tools.transmissionAnalyser.utils.transmissionAnalyser import TransmissionAnalyser


class TransmissionModelTool(TransmissionAnalyser):

    def __init__(self, transmission_test_folder, gps_cam_log_file):
        super(TransmissionModelTool, self).__init__(transmission_test_folder, gps_cam_log_file,
                                                    start_if_gps_signal=True, end_in_sync=True)
        self.model = None

    def create_model(self, model_name, bin_size=50, transmission_direction=1, vehicle_type="bike"):
        self.get_transmission_accuracy(bin_size, transmission_direction)

        transmission_model = TransmissionModel(model_name=model_name)
        self.model = transmission_model
        
        transmission_accuracy = self.get_transmission_accuracy(bin_size, transmission_direction)
        transmission_time = self.get_transmission_time(transmission_direction)

        transmission_model.transmission_accuracy = transmission_accuracy
        transmission_model.transmission_time = transmission_time

        self.model.save_model(vehicle_type)
        print("TRANSMISSION Model saved!")
        
    def use_model(self):
        pass

    def get_transmission_accuracy(self, bin_size, transmission_direction):
        outgoing = [self.evk_1_camFile_outgoing, self.evk_2_camFile_outgoing][transmission_direction - 1]
        incoming = [self.evk_2_camFile_incoming, self.evk_1_camFile_incoming][transmission_direction - 1]
        pos_receiver = [self.evk_2_gpsFile.get_positions(), self.evk_1_gpsFile.get_positions()][
            transmission_direction - 1]
        direction = ["Transmission Direction 1 (outgoing) -> 2 (incoming)",
                     "Transmission Direction 2 (outgoing) -> 1 (incoming)"][transmission_direction - 1]

        outgoing_times = outgoing.get_times()
        outgoing_pos = outgoing.get_positions()
        outgoing_ids = outgoing.get_extended_ids()

        incoming_times = incoming.get_times()
        incoming_ids = incoming.get_extended_ids()

        transmission_log = []

        for message_id, lat, lon, time in zip(outgoing_ids, outgoing_pos[0], outgoing_pos[1], outgoing_times):
            if message_id in incoming_ids:
                received = True
            else:
                received = False

            # find index for position of receiver with same time
            i = min(range(len(pos_receiver["time"])), key=lambda j: abs(pos_receiver["time"][j] - time))

            distance = distance_earth(lat, lon, pos_receiver["latitude"][i], pos_receiver["longitude"][i])
            transmission_log.append([distance, received])

        transmission_log.sort(key=lambda x: x[0])
        transmission_accuracy = {}

        # bin-size in meters
        last_bin = math.ceil(transmission_log[-1][0] / 50) * 50
        bins = np.arange(0, last_bin, bin_size)

        # convert from int32 to int for saving in json
        bins = [int(b) for b in bins]

        num_transmissions_list = []

        for bin in bins:
            num_transmissions = 0
            num_successful = 0

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
            transmission_accuracy[bin] = float(round(accuracy, 4))

        return transmission_accuracy

    def get_transmission_time(self, transmission_direction):

        outgoing = [self.evk_1_camFile_outgoing, self.evk_2_camFile_outgoing][transmission_direction - 1]
        incoming = [self.evk_2_camFile_incoming, self.evk_1_camFile_incoming][transmission_direction - 1]
        pos_receiver = [self.evk_2_gpsFile.get_positions(), self.evk_1_gpsFile.get_positions()][transmission_direction - 1]
        direction = ["Transmission Direction 1 (outgoing) -> 2 (incoming)",
                     "Transmission Direction 2 (outgoing) -> 1 (incoming)"][transmission_direction - 1]

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
                i = min(range(len(pos_receiver["time"])), key=lambda j: abs(pos_receiver["time"][j] - time))

                distance = distance_earth(lat, lon, pos_receiver["latitude"][i], pos_receiver["longitude"][i])
                transmission_time = receive_time - time

                transmission_log.append(transmission_time)

        avg_transmission_time = sum(transmission_log) / len(transmission_log)
        transmission_log = [t for t in transmission_log if 0 < t < (2 * avg_transmission_time)]
        transmission_time = sum(transmission_log) / len(transmission_log)

        return round(transmission_time, 4)