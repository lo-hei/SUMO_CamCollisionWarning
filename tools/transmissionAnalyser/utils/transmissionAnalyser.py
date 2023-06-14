import os
from pathlib import Path

from tools.transmissionAnalyser.utils.camFile import CamFile
from tools.transmissionAnalyser.utils.simple_gpsFile import SimpleGpsFile


class TransmissionAnalyser:

    def __init__(self, transmission_test_folder, gps_cam_log_file, start_if_gps_signal):
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.transmission_test_folder = transmission_test_folder
        self.gps_cam_log_file = gps_cam_log_file

        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_1\\" \
                    + gps_cam_log_file + "\\incoming_cams.csv"
        self.evk_1_camFile_incoming = CamFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_1\\" \
                    + gps_cam_log_file + "\\outgoing_cams.csv"
        self.evk_1_camFile_outgoing = CamFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_2\\" \
                    + gps_cam_log_file + "\\incoming_cams.csv"
        self.evk_2_camFile_incoming = CamFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_2\\" \
                    + gps_cam_log_file + "\\outgoing_cams.csv"
        self.evk_2_camFile_outgoing = CamFile(file_path)

        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_1\\" \
                    + gps_cam_log_file + "\\internal_GPS.csv"
        self.evk_1_gpsFile = SimpleGpsFile(file_path)
        file_path = self.root_path + "\\src\\transmissionData" + "\\" + self.transmission_test_folder + "\\EVK_2\\" \
                    + gps_cam_log_file + "\\internal_GPS.csv"
        self.evk_2_gpsFile = SimpleGpsFile(file_path)

        # start all files at the same time, when all EVKs got a GPS-Signal and are turned on
        if start_if_gps_signal:
            self.cut_all_files()

    def cut_all_files(self):
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
            f.cut_by_time(max(starting_times))
