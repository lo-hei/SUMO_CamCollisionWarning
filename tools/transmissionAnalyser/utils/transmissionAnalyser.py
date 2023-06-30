import os
from pathlib import Path

from tools.transmissionAnalyser.utils.camFile import CamFile
from tools.transmissionAnalyser.utils.simple_gpsFile import SimpleGpsFile


class TransmissionAnalyser:

    def __init__(self, transmission_test_folder, gps_cam_log_file, start_if_gps_signal, end_in_sync):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.transmission_test_folder = transmission_test_folder
        self.gps_cam_log_file = gps_cam_log_file

        if type(self.gps_cam_log_file) == list:
            gps_cam_log_file_1 = self.gps_cam_log_file[0]
            gps_cam_log_file_2 = self.gps_cam_log_file[1]


        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_1\\" \
                    + gps_cam_log_file_1 + "\\incoming_cams.csv"
        self.evk_1_camFile_incoming = CamFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_1\\" \
                    + gps_cam_log_file_1 + "\\outgoing_cams.csv"
        self.evk_1_camFile_outgoing = CamFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_2\\" \
                    + gps_cam_log_file_2 + "\\incoming_cams.csv"
        self.evk_2_camFile_incoming = CamFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_2\\" \
                    + gps_cam_log_file_2 + "\\outgoing_cams.csv"
        self.evk_2_camFile_outgoing = CamFile(file_path)

        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_1\\" \
                    + gps_cam_log_file_1 + "\\internal_GPS.csv"
        self.evk_1_gpsFile = SimpleGpsFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_2\\" \
                    + gps_cam_log_file_2 + "\\internal_GPS.csv"
        self.evk_2_gpsFile = SimpleGpsFile(file_path)

        # start all files at the same time, when all EVKs got a GPS-Signal and are turned on
        if start_if_gps_signal:
            self.cut_start_all_files()

        if end_in_sync:
            self.cut_end_all_files()

        self.sync_times()


    def cut_start_all_files(self):
        files = [self.evk_1_camFile_incoming,
                 self.evk_1_camFile_outgoing,
                 self.evk_2_camFile_incoming,
                 self.evk_2_camFile_outgoing,
                 self.evk_1_gpsFile, self.evk_2_gpsFile]

        starting_times = []
        for f in files:
            s_t = f.get_starting_time()
            starting_times.append(s_t)

        for f in files:
            f.cut_start_by_time(max(starting_times))

    def cut_end_all_files(self):
        files = [self.evk_1_camFile_incoming,
                 self.evk_1_camFile_outgoing,
                 self.evk_2_camFile_incoming,
                 self.evk_2_camFile_outgoing,
                 self.evk_1_gpsFile, self.evk_2_gpsFile]

        end_times = []
        for f in files:
            e_t = f.get_end_time()
            end_times.append(e_t)

        for f in files:
            f.cut_end_by_time(min(end_times))

    def sync_times(self):
        print(self.evk_1_camFile_incoming.print_data())