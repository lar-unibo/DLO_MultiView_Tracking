from pipy.tf import Rotation
import numpy as np


def rotation_from1to2(rot1, rot2):
    return rot1.transpose() * rot2


def rot_to_quat(rot):
    return Rotation(rot).get_quaternion()


def quat_to_rot(quat):
    return Rotation.quaternion(quat).to_numpy()

def list_quat_to_rot(quat):
    return Rotation.quaternion(quat[0], quat[1], quat[2], quat[3])

def rpy_to_rot(rpy):
    return Rotation.rpy(rpy[0], rpy[1], rpy[2]).to_numpy()


def rot_to_rpy(rot):
    return Rotation(rot).get_rpy()


def rotvec_to_rot(rotvec):
    return Rotation.rot(rotvec=rotvec, angle=np.linalg.norm(rotvec)).to_numpy()

def quat_to_rpy(quat):
    return rot_to_rpy(list_quat_to_rot(quat).to_numpy())
        