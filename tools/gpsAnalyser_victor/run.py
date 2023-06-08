from tools.gpsAnalyser_victor.gpsModelTool import GpsModelTool
from tools.gpsAnalyser_victor.gpsPlotter import GpsPlotter


def run():
    gps_cam_log_name = "GPS_TEST_32_1105462052022"

    gps_plotter = GpsPlotter(gps_cam_log_name=gps_cam_log_name)
    gps_model_tool = GpsModelTool(gps_cam_log_name=gps_cam_log_name, baseline_file="usb_r")

    interval = None

    # gps_plotter.plot_gps_track_on_map(file_name=1, interval=interval)
    gps_plotter.plot_gps_track_interpolation(file_names=["usb_1", "usb_2"], dots=True, interval=interval)
    ''' styles = [histogram, map, drift, heatmap] '''
    # gps_plotter.plot_gps_deviation(baseline=1, comparison=3, style="heatmap", interval=interval)

    # gps_model_tool.create_model(gps_file=3, model_name="GpsModels-victor-1")
    # gps_model_tool.use_model(model_name="GpsModels-victor-1", seconds_to_simulate=6)


if __name__ == '__main__':
    run()
