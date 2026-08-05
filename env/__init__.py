"""Humanoid Parkour environment: course, physics, scoring.

Shared by the referee image and the local tools so the numbers can never diverge.
"""

from .course import COURSE_LENGTH, SEGMENTS, TRACK_HALF_W, nominal_mu
from .scoring import instance_score
from .sim import (ACT_DIM, OBS_DIM, STATE_DIM, InstanceParams, InvalidAction, ParkourSim,
                  instance_spec)

__all__ = ["ACT_DIM", "COURSE_LENGTH", "OBS_DIM", "STATE_DIM", "InstanceParams", "InvalidAction",
           "ParkourSim", "SEGMENTS", "TRACK_HALF_W", "instance_score", "instance_spec",
           "nominal_mu"]
