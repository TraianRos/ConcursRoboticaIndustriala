"""
Imports from various libraries inside OS image.
THE PONY, HE COMES!
"""

from importlib import import_module
from pathlib import Path
import sys
from typing import TYPE_CHECKING


_SOURCE_PATHS = (
    "0.py_install",
    "2.sys_settings/1.OLED",
    "3.ctrl_Arm/12.sync_movement",
    "6.AI_Visual",
)

# The matching IDE paths are in .vscode/settings.json. Runtime changes to
# sys.path alone cannot give a static analyzer access to these source folders.
for _path in reversed(_SOURCE_PATHS):
    sys.path.insert(0, str(Path.cwd() / _path))

if TYPE_CHECKING:
    import cv2 as cv2
    import numpy as np
    import Arm_Lib as Arm_Lib
    from Arm_Lib import Arm_Device as Arm_Device
    import oled as oled
    import A_arm_follow as A_arm_follow
    import B_arm_ctrl as B_arm_ctrl
    import Object_recognition as Object_recognition
    import face_follow as face_follow
    import garbage_identify as garbage_identify
    import single_garbage_identify as single_garbage_identify
    from nets import CSPdarknet53_tiny as CSPdarknet53_tiny
    from nets import ious as ious
    from nets import loss as loss
    from nets import yolo4_tiny as yolo4_tiny
    from utils import utils as utils

_MODULES = {
    "cv2": "cv2",
    "np": "numpy",
    "Arm_Lib": "Arm_Lib",
    "oled": "oled",
    "A_arm_follow": "A_arm_follow",
    "B_arm_ctrl": "B_arm_ctrl",
    "Object_recognition": "Object_recognition",
    "face_follow": "face_follow",
    "garbage_identify": "garbage_identify",
    "single_garbage_identify": "single_garbage_identify",
    "CSPdarknet53_tiny": "nets.CSPdarknet53_tiny",
    "ious": "nets.ious",
    "loss": "nets.loss",
    "yolo4_tiny": "nets.yolo4_tiny",
    "utils": "utils.utils",
}

__all__ = ["Arm_Device"] + list(_MODULES)


def __getattr__(name):
    if name == "Arm_Device":
        value = import_module("Arm_Lib").Arm_Device
    elif name in _MODULES:
        value = import_module(_MODULES[name])
    else:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
