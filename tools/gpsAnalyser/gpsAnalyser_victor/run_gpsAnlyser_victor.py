import os
from pathlib import Path

from tools.gpsAnalyser.gpsAnalyser_victor.gpsFile_victor import GpsFile
from tools.gpsAnalyser.utils.gpsModelTool import GpsModelTool
from tools.gpsAnalyser.utils.gpsPlotter import GpsPlotter


def run():
    gps_cam_log_name = "GPS_TEST_31_1105462052022"
    available_files = ["usb_1", "usb_2", "usb_3", "usb_4", "usb_r"]
    file_plotter = load_data(gps_cam_log_name=gps_cam_log_name, available_files=available_files)

    gps_plotter = GpsPlotter(gps_cam_log_name=gps_cam_log_name, available_files=available_files,
                             file_plotter=file_plotter)
    gps_model_tool = GpsModelTool(gps_cam_log_name=gps_cam_log_name, available_files=available_files,
                                  file_plotter=file_plotter, baseline_file="usb_r")

    interval = None

    # gps_plotter.plot_gps(file_names=["usb_1"], interval=interval)
    # gps_plotter.plot_gps_track_on_map(file_name="usb_1", interval=interval)
    # gps_plotter.plot_gps_track_interpolation(file_names=["usb_1", "usb_r"], dots=True, interval=interval)
    gps_plotter.plot_gps_error(file_name="usb_r", interval=interval)
    ''' styles = [histogram, map, drift, heatmap] '''
    # gps_plotter.plot_gps_deviation(baseline="usb_r", comparison="usb_1", style="heatmap", interval=interval)

    # gps_model_tool.create_model(gps_file="usb_3", model_name="GpsModels-victor-1")
    # gps_model_tool.use_model(model_name="GpsModels-victor-1", seconds_to_simulate=15)


def load_data(gps_cam_log_name, available_files):
    file_plotter = {}
    current_path = Path(os.path.dirname(os.path.abspath(__file__)))
    root_path = current_path.parent.parent.__str__()

    for file in available_files:
        filename = file + "_" + str(gps_cam_log_name).split("_")[2] + "_" + str(gps_cam_log_name).split("_")[3]
        file_path = root_path + "\\src\\gpsData_victor" + "\\" + gps_cam_log_name + "\\" + filename + ".txt"
        print(file_path)
        if os.path.exists(file_path):
            print("GpsModelCreator : found ", gps_cam_log_name + "\\" + file)
            gps_file_plotter = GpsFile(file_path=file_path)
            file_plotter[file] = gps_file_plotter

    return file_plotter


if __name__ == '__main__':
    run()
