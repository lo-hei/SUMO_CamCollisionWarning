import os
from pathlib import Path

from tools.gpsAnalyser.gpsAnalyser_evk.gpsFile_evk import GpsFile
from tools.gpsAnalyser.utils.gpsModelTool import GpsModelTool
from tools.gpsAnalyser.utils.gpsPlotter import GpsPlotter


def run():
    perfect_conditions = ["GPS_Test_23_06/GpsCamLog_5",
                          "Transmission_Test_23_06/EVK_1/GpsCamLog_3",
                          "Transmission_Test_23_06/EVK_1/GpsCamLog_4",
                          "Transmission_Test_23_06/EVK_1/GpsCamLog_6"]

    difficult_conditions = ["GPS_Test_23_06/GpsCamLog_3",
                            "Transmission_Test_23_06/EVK_1/GpsCamLog_13",
                            "Transmission_Test_23_06/EVK_1/GpsCamLog_14"]

    city_conditions = ["GPS_Test_23_06/GpsCamLog_4",
                       "GPS_Test_23_06/GpsCamLog_6",
                       "GPS_Test_23_06/GpsCamLog_8",
                       "Transmission_Test_23_06/EVK_1/GpsCamLog_10",
                       "Transmission_Test_23_06/EVK_1/GpsCamLog_11"]

    bad_conditions = ["GPS_Test_23_06/GpsCamLog_7",
                      "Transmission_Test_23_06/EVK_1/GpsCamLog_9"]

    gps_cam_log_name = "GpsCamLog_204"
    available_files = ["internal_GPS", "externalACM0_GPS", "externalACM1_GPS", "externalACM2_GPS"]
    file_plotter = load_data(gps_cam_log_name=gps_cam_log_name, available_files=available_files)

    gps_plotter = GpsPlotter(gps_cam_log_name=gps_cam_log_name, available_files=available_files,
                             file_plotter=file_plotter)
    gps_model_tool = GpsModelTool(gps_cam_log_name=gps_cam_log_name, available_files=available_files,
                                  file_plotter=file_plotter, baseline_file="externalACM1_GPS")

    interval = [340, 490]

    # gps_plotter.plot_gps_track_on_map("externalACM0_GPS", interval=interval)
    # gps_plotter.plot_gps_track_interpolation(file_names=["externalACM0_GPS"], interval=interval, interpolation=False, dots=True)
    gps_plotter.plot_gps_parameter(file_names=["externalACM0_GPS", "internal_GPS"], interval=interval, type="speed_heading")
    # gps_plotter.plot_gps_error(file_name="internal_GPS", interval=interval)
    # gps_plotter.plot_gps_error(file_name="externalACM0_GPS", interval=interval)
    # gps_plotter.plot_gps_error(file_name="externalACM1_GPS", interval=interval)
    # gps_plotter.plot_gps_error(file_name="externalACM2_GPS", interval=interval)

    # gps_plotter.plot_avg_error_with_time_to_gps()

    '''
    file_plotter_list = load_data(gps_cam_log_name=bad_conditions,
                                  available_files=available_files)
    gps_plotter.plot_gps_error_v2(file_name="externalACM0_GPS", file_plotter_list=file_plotter_list)
    gps_plotter.plot_gps_error_v2(file_name="externalACM1_GPS", file_plotter_list=file_plotter_list)
    gps_plotter.plot_gps_error_v2(file_name="externalACM2_GPS", file_plotter_list=file_plotter_list)
    gps_plotter.plot_gps_error_v2(file_name="internal_GPS", file_plotter_list=file_plotter_list)
    '''

    ''' styles = [histogram, map, drift, heatmap] '''
    # gps_plotter.plot_gps_deviation(baseline="externalACM0_GPS", comparison="externalACM0_GPS", style="map")

    # gps_model_tool.create_model(gps_file="externalACM0_GPS", model_name="GpsModel-cwaTestSlow-bb")
    # gps_model_tool.create_model(gps_file="externalACM2_GPS", model_name="GpsModel-bad-handlebar")

    # gps_model_tool.use_model(model_name="GpsModel-bad-bottombracket", seconds_to_simulate=25)
    # gps_model_tool.compare_gps_and_model(file_name="externalACM0_GPS", model_name="GpsModel-bad-bottombracket",
    #                                      seconds_to_simulate=35)

def load_data(gps_cam_log_name, available_files):

    if not type(gps_cam_log_name) == list:
        gps_cam_log_name = [gps_cam_log_name]

    file_plotter_list = []

    for n in gps_cam_log_name:
        file_plotter = {}
        current_path = Path(os.path.dirname(os.path.abspath(__file__)))
        root_path = current_path.parent.parent.__str__()

        for file in available_files:
            file_path = root_path + "\\src\\cwaData\\CollisionTest_Fast\Bike-bb\\" + n + "\\" + file + ".csv"
            # file_path = root_path + "\\src\\gpsData\\" + n + "\\" + file + ".csv"
            print(file_path)
            if os.path.exists(file_path):
                print("GpsModelCreator : found ", n + "\\" + file)
                gps_file_plotter = GpsFile(file_path=file_path)
                file_plotter[file] = gps_file_plotter
        file_plotter_list.append(file_plotter)

    if len(file_plotter_list) == 1:
        return file_plotter_list[0]
    else:
        return file_plotter_list


if __name__ == '__main__':
    run()
