from tools.transmissionAnalyser.utils.transmissionModelTool import TransmissionModelTool
from tools.transmissionAnalyser.utils.transmissionPlotter import TransmissionPlotter


def run():
    transmission_test_folder = "Transmission_Test_23_06"
    gps_cam_log_file = ["GpsCamLog_13", "GpsCamLog_11"]
    start_if_gps_signal = False
    end_in_sync = False

    transmission_plotter = TransmissionPlotter(transmission_test_folder=transmission_test_folder,
                                               gps_cam_log_file=gps_cam_log_file,
                                               start_if_gps_signal=start_if_gps_signal,
                                               end_in_sync=end_in_sync)
    transmission_model_tool = TransmissionModelTool(transmission_test_folder=transmission_test_folder,
                                                    gps_cam_log_file=gps_cam_log_file)

    transmission_plotter.plot_transmissions_on_map()
    transmission_plotter.plot_transmission_success_on_distance()
    # transmission_plotter.plot_transmission_time_on_distance()
    transmission_plotter.plot_transmission_accuracy_on_distance(bin_size=50)

    transmission_model_tool.create_model(model_name="TransmissionModel-mosel",
                                         bin_size=50, transmission_direction=2, vehicle_type="bike")


if __name__ == '__main__':
    run()