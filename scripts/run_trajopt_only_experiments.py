#!/usr/bin/env python3
"""Run trajopt_only KPI trials in an already running Gazebo/ROS2 setup."""

from __future__ import annotations

import sys

from gazebo_experiment_common import build_common_parser, run_gazebo_experiment


def main() -> int:
    parser = build_common_parser(
        "trajopt_only",
        "Run Gazebo/ROS2 pick-place trials with experiment_mode=trajopt_only.",
    )
    return run_gazebo_experiment(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
