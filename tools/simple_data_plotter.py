import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def gps_error_precision():
    # define data
    x = np.array([0, 0.82, 1.1, 2, 2.2, 3.8, 4, 4.6, 5, 6, 6.25, 7.5, 8, 8.75, 10])
    y_accuracy = np.array(
        [0.769, 0.752, 0.748, 0.747, 0.779, 0.757, 0.804, 0.721, 0.750, 0.743, 0.648, 0.657, 0.585, 0.660, 0.612])
    y_f1 = np.array(
        [0.826, 0.811, 0.809, 0.806, 0.827, 0.829, 0.848, 0.803, 0.806, 0.803, 0.712, 0.715, 0.690, 0.733, 0.710])

    plt.figure(figsize=(4, 2), dpi=300)

    # create scatterplot
    plt.plot(x, y_accuracy, color="darkgreen", alpha=0.5)
    plt.plot(x, y_f1, color="yellowgreen", alpha=0.5)

    plt.scatter(x, y_accuracy, s=10, label="Accuracy", color="darkgreen")
    plt.scatter(x, y_f1, s=10, label="F1-Score", color="yellowgreen")

    plt.xlabel("GPS-Fehler in Meter")
    plt.ylabel("Genauigkeitswert")

    plt.ylim([0.40, 0.9])
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def gps_error_warning_precision():
    # define data
    x = np.array([0, 0.82, 1.1, 2, 2.2, 3.8, 4, 4.6, 5, 6, 6.25, 7.5, 8, 8.75, 10])
    y_accuracy_warning = np.array(
        [0.66, 0.7, 0.62, 0.65, 0.58, 0.61, 0.66, 0.53, 0.47, 0.53, 0.48, 0.54, 0.45, 0.58, 0.5])
    y_accuracy_collision = np.array(
        [0.78, 0.82, 0.73, 0.67, 0.67, 0.68, 0.72, 0.62, 0.63, 0.65, 0.68, 0.69, 0.65, 0.62, 0.61])

    plt.figure(figsize=(4, 2), dpi=300)

    # create scatterplot
    plt.plot(x, y_accuracy_warning, color="orange", alpha=0.5)
    plt.plot(x, y_accuracy_collision, color="darkred", alpha=0.5)

    plt.scatter(x, y_accuracy_warning, s=10, label="Accuracy Warnung", color="orange")
    plt.scatter(x, y_accuracy_collision, s=10, label="Accuracy Kollision", color="darkred")

    plt.xlabel("GPS-Fehler in Meter")
    plt.ylabel("Genauigkeitswert")

    plt.ylim([0.40, 0.9])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def gps_error_warning_times():
    # define data
    x = np.array([0, 0.82, 1.1, 2, 2.2, 3.8, 4, 4.6, 5, 6, 6.25, 7.5, 8, 8.75, 10])
    y_warning = np.array(
        [1.68, 2.67, 2.00, 2.23, 2.09, 2.51, 1.89, 2.80, 2.81, 3.06, 2.62, 2.47, 2.74, 2.87, 3.46])
    y_collision = np.array(
        [0.65, 0.64, 0.93, 0.65, 0.48, 0.94, 0.60, 0.85, 0.68, 0.50, 0.98, 0.58, 1.09, 1.15, 1.54])

    plt.figure(figsize=(3, 3), dpi=300)

    # create scatterplot
    plt.plot(x, y_warning, color="orange", alpha=0.5)
    plt.plot(x, y_collision, color="darkred", alpha=0.5)

    plt.scatter(x, y_warning, s=10, label="Warnung", color="orange")
    plt.scatter(x, y_collision, s=10, label="Kollision", color="darkred")

    # calculate equation for trendline
    z_accuracy = np.polyfit(x, y_warning, 1)
    z_f1 = np.polyfit(x, y_collision, 1)
    p_accuracy = np.poly1d(z_accuracy)
    p_f1 = np.poly1d(z_f1)

    # add trendline to plot
    plt.plot(x, p_accuracy(x), color="orange", ls="--", alpha=0.5)
    plt.plot(x, p_f1(x), color="darkred", ls="--", alpha=0.5)

    plt.xlabel("GPS-Fehler in Meter")
    plt.ylabel("Vorwarnzeit in Sekunden")

    plt.legend()
    plt.tight_layout()
    plt.show()


def cam_freq_precision():
    # define data
    x = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
    y_accuracy = np.array(
        [0.405, 0.418, 0.640, 0.748, 0.764, 0.653, 0.610])
    y_f1 = np.array(
        [0.214, 0.170, 0.673, 0.817, 0.825, 0.768, 0.721])

    fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=200)

    # create scatterplot
    ax1.plot(x, y_accuracy, color="darkgreen", alpha=0.5)
    ax1.plot(x, y_f1, color="yellowgreen", alpha=0.5)

    ax1.scatter(x, y_accuracy, s=10, label="Accuracy", color="darkgreen")
    ax1.scatter(x, y_f1, s=10, label="F1-Score", color="yellowgreen")

    ax1.set_xlabel("Übertragungsfrequenz in Hz")
    ax1.set_ylabel("Genauigkeitswert")

    ax1.set_ylim([0.15, 0.9])
    # convert x-axis to Logarithmic scale
    ax1.set_xscale("log")
    ax1.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def cam_freq_warning_precision():
    # define data
    x = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
    y_warning_accuracy = np.array(
        [0.53, 0.63, 0.66, 0.5, 0.54, 0.42, 0.53])
    y_collision_accuracy = np.array(
        [0.67, 0.61, 0.67, 0.67, 0.63, 0.57, 0.74])

    fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=200)

    # create scatterplot
    ax1.plot(x, y_warning_accuracy, color="orange", alpha=0.5)
    ax1.plot(x, y_collision_accuracy, color="darkred", alpha=0.5)

    ax1.scatter(x, y_warning_accuracy, s=10, label="Accuracy Warnung", color="orange")
    ax1.scatter(x, y_collision_accuracy, s=10, label="Accuracy Kollision", color="darkred")

    ax1.set_xlabel("Übertragungsfrequenz in Hz")
    ax1.set_ylabel("Genauigkeitswert")

    ax1.set_ylim([0.15, 0.9])
    # convert x-axis to Logarithmic scale
    ax1.set_xscale("log")
    ax1.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def cam_freq_warning_times():
    # define data
    x = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
    y_warning = np.array(
        [3.00, 3.00, 2.05, 2.38, 3.07, 3.30, 3.53])
    y_collision = np.array(
        [0.00, 0.00, 0.42, 0.44, 1.03, 0.65, 0.70])

    fig1, ax1 = plt.subplots(figsize=(3, 3), dpi=200)

    # create scatterplot
    ax1.plot(x, y_warning, color="orange", alpha=0.5)
    ax1.plot(x, y_collision, color="darkred", alpha=0.5)

    ax1.scatter(x, y_warning, s=10, label="Warnung", color="orange")
    ax1.scatter(x, y_collision, s=10, label="Kollision", color="darkred")

    # calculate equation for trendline
    z_accuracy = np.polyfit(x, y_warning, 1)
    z_f1 = np.polyfit(x, y_collision, 1)
    p_accuracy = np.poly1d(z_accuracy)
    p_f1 = np.poly1d(z_f1)

    # add trendline to plot
    ax1.plot(x, p_accuracy(x), color="orange", ls="--", alpha=0.5)
    ax1.plot(x, p_f1(x), color="darkred", ls="--", alpha=0.5)

    ax1.set_xlabel("Übertragungsfrequenz in Hz")
    ax1.set_ylabel("Vorwarnzeit in Sekunden")

    # convert x-axis to Logarithmic scale
    ax1.set_xscale("log")
    ax1.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend()
    plt.tight_layout()
    plt.show()


def cam_range_precision():
    # define data
    x = np.array([10, 15, 25, 50, 75, 100, 200, 500])
    y_accuracy = np.array(
        [0.640, 0.760, 0.798, 0.822, 0.804, 0.771, 0.843, 0.788])
    y_f1 = np.array(
        [0.509, 0.782, 0.833, 0.864, 0.853, 0.806, 0.892, 0.845])

    fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=200)

    # create scatterplot
    ax1.plot(x, y_accuracy, color="darkgreen", alpha=0.5)
    ax1.plot(x, y_f1, color="yellowgreen", alpha=0.5)

    ax1.scatter(x, y_accuracy, s=10, label="Accuracy", color="darkgreen")
    ax1.scatter(x, y_f1, s=10, label="F1-Score", color="yellowgreen")

    ax1.set_xlabel("Übertragungsreichweite in Meter")
    ax1.set_ylabel("Genauigkeitswert")

    ax1.set_ylim([0.35, 0.9])
    # convert x-axis to Logarithmic scale
    ax1.set_xscale("log")
    ax1.set_xlim([8, 600])
    ax1.set_xticks([10, 15, 25, 50, 75, 100, 200, 500])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def cam_range_warning_precision():
    # define data
    x = np.array([10, 15, 25, 50, 75, 100, 200, 500])
    y_warning_accuracy = np.array(
        [0.62, 0.6, 0.67, 0.65, 0.67, 0.66, 0.64, 0.59])
    y_collision_accuracy = np.array(
        [0.64, 0.82, 0.82, 0.72, 0.73, 0.73, 0.72, 0.75])

    fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=200)

    # create scatterplot
    ax1.plot(x, y_warning_accuracy, color="orange", alpha=0.5)
    ax1.plot(x, y_collision_accuracy, color="darkred", alpha=0.5)

    ax1.scatter(x, y_warning_accuracy, s=10, label="Accuracy Warnung", color="orange")
    ax1.scatter(x, y_collision_accuracy, s=10, label="Accuracy Kollision", color="darkred")

    ax1.set_xlabel("Übertragungsreichweite in Meter")
    ax1.set_ylabel("Genauigkeitswert")

    ax1.set_ylim([0.35, 0.9])
    # convert x-axis to Logarithmic scale
    ax1.set_xscale("log")
    ax1.set_xlim([8, 600])
    ax1.set_xticks([10, 15, 25, 50, 75, 100, 200, 500])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def cam_range_warning_times():
    # define data
    x = np.array([10, 15, 25, 50, 75, 100, 200, 500])
    y_warning = np.array(
        [0.05, 0.68, 1.22, 1.36, 2.24, 2.34, 2.64, 2.52])
    y_collision = np.array(
        [0.30, 0.83, 0.88, 0.79, 0.74, 0.75, 0.84, 0.64])

    fig1, ax1 = plt.subplots(figsize=(3, 3), dpi=200)

    # create scatterplot
    ax1.plot(x, y_warning, color="orange", alpha=0.5)
    ax1.plot(x, y_collision, color="darkred", alpha=0.5)

    ax1.scatter(x, y_warning, s=10, label="Warnung", color="orange")
    ax1.scatter(x, y_collision, s=10, label="Kollision", color="darkred")

    # calculate equation for trendline
    z_accuracy = np.polyfit(x, y_warning, 1)
    z_f1 = np.polyfit(x, y_collision, 1)
    p_accuracy = np.poly1d(z_accuracy)
    p_f1 = np.poly1d(z_f1)

    # add trendline to plot
    ax1.plot(x, p_accuracy(x), color="orange", ls="--", alpha=0.5)
    ax1.plot(x, p_f1(x), color="darkred", ls="--", alpha=0.5)

    ax1.set_xlabel("Übertragungsreichweite in Meter")
    ax1.set_ylabel("Vorwarnzeit in Sekunden")

    # convert x-axis to Logarithmic scale
    ax1.set_xscale("log")
    ax1.set_xlim([8, 600])
    ax1.set_xticks([10, 15, 25, 50, 100, 200, 500])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend()
    plt.tight_layout()
    plt.show()


def cam_sensor_accuracy():
    X = ['Kein Sensor', 'IMU', 'Tachometer', 'Kompass  ', '    IMU&Tachometer']
    accuracy = [0.781, 0.788, 0.794, 0.772, 0.802]
    f1_score = [0.827, 0.837, 0.842, 0.841, 0.848]
    accuracy_warning = [0.61, 0.69, 0.62, 0.59, 0.59]
    accuracy_collision = [0.67, 0.75, 0.69, 0.65, 0.66]

    time_warning = [2, 3, 2, 3, 4]
    time_collision = [2, 3, 2, 3, 4]

    plt.figure(figsize=(7, 3), dpi=200)
    X_axis = np.arange(len(X))

    plt.axhline(accuracy[0], ls="--", xmin=0.1, color="darkgreen", alpha=0.5)
    plt.axhline(f1_score[0], ls="--", xmin=0.05, color="yellowgreen", alpha=0.5)
    plt.axhline(accuracy_warning[0], ls="--", xmin=0.12, color="orange", alpha=0.5)
    plt.axhline(accuracy_collision[0], ls="--", xmin=0.17, color="darkred", alpha=0.5)

    plt.bar(X_axis - 0.1, accuracy, 0.15, label='Accuracy', color="darkgreen", alpha=1)
    plt.bar(X_axis - 0.25, f1_score, 0.15, label='F1-Score', color="yellowgreen", alpha=1)
    plt.bar(X_axis + 0.1, accuracy_warning, 0.15, label='Accuracy Warnung', color="orange", alpha=1)
    plt.bar(X_axis + 0.25, accuracy_collision, 0.15, label='Accuracy Kollision', color="darkred", alpha=1)


    plt.ylim([0.4, 0.9])
    plt.xticks(X_axis, X)
    plt.xlabel("Zusätzliche Sensoren")
    plt.ylabel("Genauigkeitswert")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


# gps_error_precision()
# gps_error_warning_times()
# gps_error_warning_precision()

# cam_freq_precision()
# cam_freq_warning_precision()
# cam_freq_warning_times()

# cam_range_precision()
# cam_range_warning_precision()
# cam_range_warning_times()

cam_sensor_accuracy()