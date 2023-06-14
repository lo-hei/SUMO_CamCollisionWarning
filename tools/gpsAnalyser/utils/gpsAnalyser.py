import os
from pathlib import Path


class GpsAnalyser:

    def __init__(self, gps_cam_log_name, available_files, file_plotter):

        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_path = current_path.parent.parent.__str__()

        self.gps_cam_log_name = gps_cam_log_name
        self.available_files = available_files
        self.file_plotter = file_plotter