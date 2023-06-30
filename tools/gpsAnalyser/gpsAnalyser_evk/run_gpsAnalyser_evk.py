import os
from pathlib import Path

from tools.gpsAnalyser.gpsAnalyser_evk.gpsFile_evk import GpsFile
from tools.gpsAnalyser.utils.gpsModelTool import GpsModelTool
from tools.gpsAnalyser.utils.gpsPlotter import GpsPlotter


def run():
    gps_cam_log_name = "GPS_Test_23_06/GpsCamLog_7"
    available_files = ["internal_GPS", "externalACM0_GPS", "externalACM1_GPS", "externalACM2_GPS"]
    file_plotter = load_data(gps_cam_log_name=gps_cam_log_name, available_files=available_files)

    gps_plotter = GpsPlotter(gps_cam_log_name=gps_cam_log_name, available_files=available_files,
                             file_plotter=file_plotter)
    gps_model_tool = GpsModelTool(gps_cam_log_name=gps_cam_log_name, available_files=available_files,
                                  file_plotter=file_plotter, baseline_file="externalACM1_GPS")

    interval = None

    # gps_plotter.plot_gps_track_on_map("externalACM0_GPS", interval=interval)
    # gps_plotter.plot_gps_track_interpolation(file_names=["externalACM1_GPS", "externalACM2_GPS"], interval=interval, interpolation=True, dots=True)
    # gps_plotter.plot_gps_error(file_name="externalACM0_GPS", interval=interval)
    # gps_plotter.plot_gps_error(file_name="externalACM1_GPS", interval=interval)
    # gps_plotter.plot_gps_error(file_name="externalACM2_GPS", interval=interval)
    gps_plotter.plot_gps_error(file_name="internal_GPS", interval=interval)
    ''' styles = [histogram, map, drift, heatmap] '''
    # gps_plotter.plot_gps_deviation(baseline="externalACM0_GPS", comparison="externalACM0_GPS", style="map")

    # gps_model_tool.create_model(gps_file="externalACM2_GPS", model_name="GpsModels-internal")
    # gps_model_tool.use_model(model_name="stored_GpsModels-internal", seconds_to_simulate=50)


def load_data(gps_cam_log_name, available_files):
    file_plotter = {}
    current_path = Path(os.path.dirname(os.path.abspath(__file__)))
    root_path = current_path.parent.parent.__str__()

    for file in available_files:
        file_path = root_path + "\\src\\gpsData" + "\\" + gps_cam_log_name + "\\" + file + ".csv"
        print(file_path)
        if os.path.exists(file_path):
            print("GpsModelCreator : found ", gps_cam_log_name + "\\" + file)
            gps_file_plotter = GpsFile(file_path=file_path)
            file_plotter[file] = gps_file_plotter

    return file_plotter


if __name__ == '__main__':
    run()
