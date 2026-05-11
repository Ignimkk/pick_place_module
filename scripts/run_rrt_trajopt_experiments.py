#!/usr/bin/env python3
"""Run rrt_trajopt KPI trials in an already running Gazebo/ROS2 setup."""

from __future__ import annotations

import sys

from gazebo_experiment_common import build_common_parser, run_gazebo_experiment


def main() -> int:
    parser = build_common_parser(
        "rrt_trajopt",
        "Run Gazebo/ROS2 pick-place trials with experiment_mode=rrt_trajopt.",
    )
    return run_gazebo_experiment(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
