from tools.gpsAnalyser.gpsModelTool import GpsModelTool
from tools.gpsAnalyser.gpsPlotter import GpsPlotter


def run():
    gps_cam_log_name = "GpsCamLog_13"
    gps_plotter = GpsPlotter(gps_cam_log_name=gps_cam_log_name)
    gps_model_tool = GpsModelTool(gps_cam_log_name=gps_cam_log_name,
                                        baseline_file="externalACM0_GPS")

    # gps_plotter.plot_gps_track_on_map("internal_GPS")
    gps_plotter.plot_gps_track_interpolation(file_names=["internal_GPS"], dots=True)
    ''' styles = [histogram, map, drift, heatmap] '''
    gps_plotter.plot_gps_deviation(baseline="externalACM0_GPS", comparison="externalACM0_GPS", style="map")

    # gps_model_tool.create_model(gps_file="externalACM0_GPS", model_name="GpsModels-internal")
    # gps_model_tool.use_model(model_name="stored_GpsModels-internal", seconds_to_simulate=15)


if __name__ == '__main__':
    run()
