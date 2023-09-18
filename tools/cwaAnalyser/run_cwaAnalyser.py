import numpy as np

from tools.cwaAnalyser.cwa_plotter import CwaPlotter


def run():
    cwa_data_path = "CollisionTest_Slow\\Bike-handlebar\\GpsCamLog_199"
    cwa_plotter = CwaPlotter(cwa_data_path=cwa_data_path)

    # cwa_plotter.plot_prob_collision()
    cwa_plotter.evaluate_tests()

    confusion_matrix = np.asarray([[8, 7, 1], [2, 3, 5], [0, 0, 4]], dtype=int)
    warning_times = [5] * 30
    collision_times = [2] * 30

    # cwa_plotter.plot_cwa_evaluation(confusion_matrix, warning_times, collision_times)

if __name__ == '__main__':
    run()