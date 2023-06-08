import os
from pathlib import Path

import tools.gpsAnalyser_victor.utils.gpsFile as gpsFile


class GpsAnalyser:

    def __init__(self, gps_cam_log_name):

        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.gps_cam_log_name = gps_cam_log_name
        self.available_files = ["usb_1", "ubs_2", "ubs_3", "usb_4", "ubs_r"]
        self.file_plotter = self.load_data()

    def load_data(self):
        file_plotter = {}

        for file in self.available_files:
            filename = file + "_" + str(self.gps_cam_log_name).split("_")[2] + "_" + str(self.gps_cam_log_name).split("_")[3]
            file_path = self.root_path + "\\src\\gpsData_victor" + "\\" + self.gps_cam_log_name + "\\" + filename + ".txt"
            print(file_path)
            if os.path.exists(file_path):
                print("GpsModelCreator : found ", self.gps_cam_log_name + "\\" + file)
                gps_file_plotter = gpsFile.GpsFile(file_path=file_path)
                file_plotter[file] = gps_file_plotter

        return file_plotter