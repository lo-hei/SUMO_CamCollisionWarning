from tools.transmissionAnalyser.utils.transmissionModelTool import TransmissionModelTool
from tools.transmissionAnalyser.utils.transmissionPlotter import TransmissionPlotter


def run():
    transmission_test_folder = "TransmissionTest_1"
    gps_cam_log_file = "GpsCamLog_4"
    start_if_gps_signal = True

    transmission_plotter = TransmissionPlotter(transmission_test_folder=transmission_test_folder,
                                               gps_cam_log_file=gps_cam_log_file,
                                               start_if_gps_signal=start_if_gps_signal)
    transmission_model_tool = TransmissionModelTool(transmission_test_folder=transmission_test_folder,
                                                    gps_cam_log_file=gps_cam_log_file)

    # transmission_plotter.plot_transmissions_on_map()
    # transmission_plotter.plot_transmission_success_on_distance()
    # transmission_plotter.plot_transmission_time_on_distance()
    # transmission_plotter.plot_transmission_accuracy_on_distance(bin_size=50)

    transmission_model_tool.create_model(model_name="TransmissionModel-preTest",
                                         bin_size=50, transmission_direction=2, vehicle_type="car")


if __name__ == '__main__':
    run()