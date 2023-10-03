# About The Project

This project was made by **Lorenz Heinemann** for his master-thesis at 
[Universität Leipzig](https://www.uni-leipzig.de/) in cooperation with CANYON Bicycle GmbH

Beginning of work: *April 2023* \
Project finished: *September 2023*

> **This project implements the following things:**
> * SUMO-Extension to simulate V2X for Vehicles with CAM Messages
> * GPS-Model to simulate a real GPS-Behavior
> * Transmission-Model to simulate a real Transmission-Behavior
> * Multiple Collision-Warning-Algorithm which can interact with SUMO
> * Tool to Analyse GPS-Data
> * Tool to Analyse Transmission-Data
> * Tool to Analyse CWA-Data
> * Tool for plotting relevant Plots like confusion Matrix

---

## Built With

* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://scipy.org/)

These packages are just the most important ones. The full list of all required packages to install can be found
in the `requirements.txt` \
Later in this README it will be described how to install all of them at once.

---

# Getting Started

In this part you will learn how to set up the project correctly.
You can run it completly from the console, or an IDE.
## Prerequisites

Create a new project or if you won't use an IDE please just create a new folder somewhere on your Computer


### pip

Please make sure that [pip](https://pip.pypa.io/en/stable/) is installed on your machine.
To check it, write the following command in the console
```
pip --version
```
The project was build on version 20.3.3. higher versions should bw okay. To upgrade please write
```
python -m pip install --upgrade pip
```
If pip is not installed, please visit this [tutorial for Windows](https://phoenixnap.com/kb/install-pip-windows)
or [this one for Linux](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)

1. Clone the project from this [Git-repo](https://www.iat.uni-leipzig.de/git/bachelorarbeit-heinemann.git)
2. install all the required packages
   ```
   pip install -r requirements.txt
   ```

### sumo

Please download and install SUMO from the official [Website](https://sumo.dlr.de)

--- 

# Usage

 ### SUMO-Extension to simulate V2X for Vehicles with CAM Messages

To start SUMO, just run the `runner.py`. All parameter can be changed there

<br>

 ### GPS-Model to simulate a real GPS-Behavior

Models are stored in `/models/stored_GpsModels` and can be created with the Model-Create-Tool 
in `/tools/gpsAnalyser/gpsAnalyser_evk`. Here you can find a model-create-methode with all settings needed.

<br>

 ### Transmission-Model to simulate a real Transmission-Behavior

Models are stored in `/models/stored_TransmissionModels` and can be created with the Model-Create-Tool 
in `/tools/gpsAnalyser/transmissionAnalyser_evk`. Here you can find a model-create-methode with all settings needed.

<br>

 ### Multiple Collision-Warning-Algorithm which can interact with SUMO

Change the used CWA by changing the CWA-Parameter in `runner.py`

<br>

 ### Tool to Analyse GPS-Data

Can be found in `/tools/gpsAnalyser/gpsAnalyser_evk/run_pgsAnalyser_evk.py`. Multiple methods for plotting can be found here. All possible 
options can be seen as comments. You can change the used Log-File with the variable _gps_cam_log_name_.

<br>

 ### Tool to Analyse Transmission-Data

Can be found in `/tools/gpsAnalyser/transmissionAnalyser_evk/run_transmissionAnalyser.py`. Multiple methods for plotting can be found here. All possible 
options can be seen as comments. You can change the used Log-File with the variable _gps_cam_log_name_.

<br>

 ### Tool to Analyse CWA-Data

Can be found in `/tools/gpsAnalyser/cwaAnalyser/run_cwaAnalyser.py`. Multiple methods for plotting can be found here. All possible 
options can be seen as comments. You can change the used Log-File with the variable _gps_cam_log_name_.

<br>

 ### Tool for plotting relevant Plots like confusion Matrix

Can be found in `/tools/simple_data_plotter.py`. Multiple methods for plotting can be found here. All possible 
options can be seen as comments. You can change the used Log-File with the variable _gps_cam_log_name_.

---

<!-- CONTACT -->
# Contact

Lorenz Heinemann - **lorenz.heinemann@web.de**