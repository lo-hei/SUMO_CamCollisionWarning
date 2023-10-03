import numpy as np

from tools.cwaAnalyser.cwa_plotter import CwaPlotter


def run():
    cwa_data_path = "CollisionTest_HighFrequency\\Bike-handlebar\\GpsCamLog_207"
    cwa_plotter = CwaPlotter(cwa_data_path=cwa_data_path)

    cwa_plotter.plot_prob_collision()
    # cwa_plotter.evaluate_tests()

    # confusion_matrix = np.asarray([[8, 10, 1], [2, 0, 8], [0, 0, 1]], dtype=int)
    confusion_matrix = np.asarray([[22, 11, 4], [3, 13, 9], [8, 12, 18]], dtype=int)

    warning_times = [5] * 30
    collision_times = [2] * 30

    # cwa_plotter.plot_cwa_evaluation(confusion_matrix, warning_times, collision_times)

if __name__ == '__main__':
    run()