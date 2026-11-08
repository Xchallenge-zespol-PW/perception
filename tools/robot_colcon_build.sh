#!/usr/bin/env bash
# Build g1_perception inside the robot humble container (cwd workspace = /ws).
# Invoked by deploy_robot.sh via:
#   ~/g1_humble/humble.sh run bash /ws/src/perception/tools/robot_colcon_build.sh
# so we avoid bash -lc quoting that humble.sh strips.
set -eo pipefail
# setup.bash touches unset vars (e.g. AMENT_TRACE_SETUP_FILES); nounset breaks it.
set +u
source /opt/ros/humble/setup.bash
set -u
cd /ws
# ~/.local often has setuptools/packaging mismatch
# (canonicalize_version(..., strip_trailing_zero=...) TypeError). Build with
# distro packages; runtime still uses --user installs.
export PYTHONNOUSERSITE=1
# --packages-up-to also builds xchallenge_msgs from /ws/src (deploy_robot.sh).
colcon build --symlink-install --packages-up-to g1_perception
