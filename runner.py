from __future__ import absolute_import
from __future__ import print_function

import math
import os
import sys
import optparse
import random
import time

from time import sleep
from tqdm import tqdm

from simulationClasses.CollisionWarningAlgorithm.dangerZonesCWA import DangerZonesCWA
from simulationClasses.CollisionWarningAlgorithm.dangerZonesCWA_production import DangerZonesCWA_production
from simulationClasses.CollisionWarningAlgorithm.dangerZonesCWA_v2 import DangerZonesCWA_v2
from simulationClasses.CollisionWarningAlgorithm.radiusCWA import RadiusCWA
from simulationClasses.CollisionWarningAlgorithm.radiusInterpolationCWA import RadiusInterpolateCWA
from simulationClasses.cwaEvaluator import CwaEvaluator
from simulationClasses.routeManager import RouteManager
from simulationClasses.simulationManager import SimulationManager

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def run():
    SPEED_CONTROL = False
    GUI = False

    step_length = 0.1

    ''' 
    evaluation_mode 
    0: NO EVALUATION
    1: PLOT_VEHICLE_PATHS
    2: PLOT_DISTANCE_BETWEEN_VEHICLES
    3: PLOT_DISTANCE_BETWEEN_CWA
    4: PLOT_CWA_PARAMETER
    5: PLOT_POC_ERROR
    6: ANALYSE_CWA_PARAMETER
    7: EVALUATE_CWA
    8: EVALUATE_TRANSMISSION
    '''
    runs = 1
    evaluation_mode = [4]
    evaluator = CwaEvaluator(runs=runs, cwa=DangerZonesCWA_v2, evaluation_mode=evaluation_mode,
                             title="small radius")

    simulationManager = SimulationManager(step_length=step_length,
                                          speed_controller=SPEED_CONTROL, evaluator=evaluator,
                                          gps_model="GpsModel-perfect-handlebar",
                                          transmission_model="TransmissionModel-mosel")

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if not GUI:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "simulationData/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml",
                 "--step-length", str(step_length),
                 "--collision-output", "collisions.txt",
                 "--collision.mingap-factor", "0",
                 "--collision.action", "remove",
                 "--collision.check-junctions",
                 "--time-to-teleport", "-1"])

    """execute the TraCI control loop"""
    p_bar = tqdm(range(runs))

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()
        simulationManager.simulation_step()

        current_run = math.ceil(simulationManager.time / simulationManager.evaluator.route_manager.gap_between_repeats) - 1
        p_bar.n = current_run
        p_bar.refresh()

    evaluator.evaluate()

    traci.close()
    sys.stdout.flush()


# this is the main entry point of this script
if __name__ == '__main__':
    run()