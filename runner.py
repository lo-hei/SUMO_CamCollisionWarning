#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2023 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random

from simulation_classes.routeManager import RouteManager
from simulation_classes.simulationManager import SimulationManager

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def generate_routefile():
    routeManager = RouteManager()
    get_bike_left_car_straight = routeManager.get_bike_straight_car_straight(repeats=1)

def run():
    gui = True

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if not gui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    collisions = 0

    # first, generate the route file for this simulation
    generate_routefile()
    step_length = 0.01

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml",
                 "--step-length", str(step_length),
                 "--collision-output", "collisions.txt",
                 "--collision.action", "remove",
                 "--collision.check-junctions"])

    """execute the TraCI control loop"""
    step = 0
    simulationManager = SimulationManager(step_length=step_length)

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()
        simulationManager.simulationStep()

        step += 1

    collisions += len(traci.simulation.getCollisions())

    traci.load([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml",
                 "--step-length", str(step_length),
                 "--collision.mingap-factor", "0",
                 "--collision.action", "remove",
                 "--collision.check-junctions"])

    traci.close()
    sys.stdout.flush()


# this is the main entry point of this script
if __name__ == '__main__':
    run()
