#!/usr/bin/env python
# -*- coding: utf-8 -*-

import skfuzzy as fuzz
from numpy.ma.core import argmin
from tensorboardX import SummaryWriter
import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data
import open3d as o3d
import random
import math
from collections import namedtuple
from attrdict import AttrDict
from scipy.spatial.transform import Rotation as R
import symbol
from sympy import *
from typing import Any, Dict, Union
sin = math.sin
from scipy.spatial.transform import Rotation
import transforms3d as tfs
import numpy as np
from IPython import display
import pybullet as p
import math
import time
import cv2
import sys
from spatialmath import SE3
import spatialmath as sm
sys.path.append('../../../imitation_learning_idp3')
from imitation_learning_idp3.arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from imitation_learning_idp3.env.pybulletsim.pybullet_planning import (
    get_joint_limits,
    get_max_velocity,
    get_movable_joints,
    get_joint_positions,
    disconnect,
)
def fix_center_rotation(end_pos, end_orn, relative_offset, relative_euler, dy_M=0.055):
    """
       目的：固定点旋转
       Arguments:
       - end_pos: len=3, 该 link 的在世界坐标系的位置
       - end_orn: len=4, 该 link 的在世界坐标系的姿态 (x, y, z, w)
       - relative_offset 该 link下的相对移动 list of 3
       - relative_euler  该 link下的旋转  list of 3
       用法：eeink_link_next_Rotation_matrix = fix_center_rotation(self.current_pos, self.current_orie, [-0.05,0,0], [0, 0.3, 0])
       -[-0.05,0,0] eelink与固定轴端面的距离(X,Y,Z)
       - [0, 0.3, 0] 绕固定轴端面的旋转欧拉角(x,y,z)

       Returns:
       - eeink_link_next_Rotation_matrix: shape=(4, 4), transform matrix, represents this link next pose in world frame
       """
    # 前置变换：先平移eelink

    # peg_link_Rotation_matrix=relative_pos_and_ore_form_world(end_pos, end_orn, relative_offset, [0, 0, 0])#只平移不旋转
    # peg_link_Quaternion = Rotation.from_matrix(peg_link_Rotation_matrix[:3, :3]).as_quat()
    # peg_link_pos = peg_link_Rotation_matrix[:3, -1]

    # 1.先将eelink变换到peg末端坐标系,再平移相对位置
    peg_link_Rotation_matrix = relative_pos_and_ore_form_world(end_pos, end_orn,
                                                               [relative_offset[0] + dy_M, relative_offset[1] + 0,
                                                                relative_offset[2] + 0], [0, 0, 0])  #
    peg_link_Quaternion = Rotation.from_matrix(peg_link_Rotation_matrix[:3, :3]).as_quat()
    peg_link_pos = peg_link_Rotation_matrix[:3, -1]

    # 2.eelink在peg坐标系做完旋转
    eeink_link_next_Rotation_matrix = relative_pos_and_ore_form_world(peg_link_pos, peg_link_Quaternion, [0, 0, 0],
                                                                      relative_euler)
    # 3.通过相对坐标系平移变换，将peg坐标下的eelink坐标映射回真实eelink坐标
    eeink_link_next_Rotation_matrix = relative_pos_and_ore_form_world(eeink_link_next_Rotation_matrix[:3, -1],
                                                                      Rotation.from_matrix(
                                                                          eeink_link_next_Rotation_matrix[:3,
                                                                          :3]).as_quat(),
                                                                      [-dy_M, 0, 0], [0, 0, 0])

    return eeink_link_next_Rotation_matrix

def relative_pos_and_ore_form_world(end_pos, end_orn, relative_offset, relative_euler):
    """
    目的：将该link下的相对移动和转动映射到绝对坐标下(当然，前两个值的父坐标系不是绝对坐标系，则以相关坐标系为准）
    Arguments:
    - end_pos: len=3, 该 link 的在世界坐标系的位置
    - end_orn: len=4, 该 link 的在世界坐标系的姿态 (x, y, z, w)
    - relative_offset 该 link下的相对移动 list of 3
    - relative_euler  该 link下的旋转  list of 3
    - Rotation.from_euler('XYZ', move_euler).as_matrix() 大写是内旋动轴旋转，小写相反

    注意：请注意该函数先旋转后移动，务必注意自己的变换要求！

    Returns:
    - wcT: shape=(4, 4), transform matrix, represents this link pose in world frame
    """

    end_orn = R.from_quat(end_orn).as_matrix()
    wcT = np.eye(4)
    # wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos #注意：务必注意自己的变换要求！
    fg = Rotation.from_euler('xyz', relative_euler).as_matrix()
    wcT[:3, :3] = np.matmul(end_orn[:3, :3], fg)
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos

    return wcT

def get_quaternion_from_matrix(matrix, isprecise=False):
    "0->w,1->x,2->y,3->z"
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def quaternion_to_6d(quat: np.ndarray, order: str = 'xyzw') -> np.ndarray:
    """
    使用 scipy 将四元数转换为6D表示（旋转矩阵的前两列）。
    参数:
        quat: 四元数，形状为 [..., 4] (xyzw 或 wxyz 顺序)。
        order: 四元数顺序，'xyzw' (默认) 或 'wxyz'。
    返回:
        6D向量，形状为 [..., 6]。
    """
    if order == 'wxyz':
        quat = np.roll(quat, shift=-1, axis=-1)  # wxyz -> xyzw
    # 使用 scipy 的 Rotation 类直接计算旋转矩阵
    rot_matrix = Rotation.from_quat(quat).as_matrix()
    # 取前两列并展平

    return rot_matrix[..., :2].reshape(*rot_matrix.shape[:-2], 6)

def _6d_to_quaternion(sixd: np.ndarray) -> np.ndarray:
    rot_matrix = sixd.reshape(-1, 3, 2)
    r3 = np.cross(rot_matrix[..., 0], rot_matrix[..., 1])
    full_matrix = np.concatenate([rot_matrix, r3[..., None]], axis=-1)

    return Rotation.from_matrix(full_matrix).as_quat()

class DebugAxes(object):
    """
    可视化某个局部坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    """

    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos).reshape(3)

        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


class Interpolation:

    def shift_array(self, arr):
        if len(arr) <= 1:
            return arr
        first_ele = arr.pop(0)
        arr.append(first_ele)

        return arr

    def cal_planner(self, t0, R0, t1, R1, time):
        position_parameter = LinePositionParameter(t0, t1)
        attitude_parameter = OneAttitudeParameter(R0, R1)
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        trajectory_planner = TrajectoryPlanner(trajectory_parameter)
        return trajectory_planner

    def get_quaternion_from_matrix(self, matrix, isprecise=False):
        "0->w,1->x,2->y,3->z"
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                          [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                          [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                          [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    def mutiPointInterpolation(self, target_positions, target_quaternions, total_ts):
        self._timeStep = 120
        # 处理后的target_positions和target_quaternions
        target_positions_interpolate = []
        target_quaternions_interpolate = []

        planner_array = []  # 储存处理后的路点

        # 处理第一个点
        homoTransMatrix1 = np.eye(4)
        homoTransMatrix1[:3, :3] = R.from_quat(target_quaternions[0]).as_matrix()
        homoTransMatrix1[:3, 3] = target_positions[0]
        T0 = SE3(homoTransMatrix1)
        t0 = T0.t
        R0 = sm.SO3(T0.R)

        last_t = t0.copy()
        last_R = R0.copy()

        planner = self.cal_planner(t0, R0, last_t, last_R, total_ts[1])
        planner_array.append(planner)

        for i in range(len(target_positions) - 1):
            # 第i段轨迹

            ## waypoint i+1
            homoTransMatrix2 = np.eye(4)
            homoTransMatrix2[:3, :3] = R.from_quat(target_quaternions[i + 1]).as_matrix()
            homoTransMatrix2[:3, 3] = target_positions[i + 1]

            T2 = SE3(homoTransMatrix2)
            t2 = T2.t
            R2 = sm.SO3(T2.R)
            planner = self.cal_planner(last_t, last_R, t2, R2, total_ts[i + 2])
            planner_array.append(planner)
            last_t = t2.copy()
            last_R = R2.copy()

        time_array = np.array(total_ts)
        total_time = np.sum(time_array)

        time_step_num = 100  # 插值点个数
        time_step_num = round(total_time * self._timeStep) + 1
        times = np.linspace(0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)

        for i, timei in enumerate(times):
            for j in range(len(time_cumsum)):
                if timei < time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    target_t = planner_interpolate.t
                    target_quat = self.get_quaternion_from_matrix(planner_interpolate.R)
                    target_quat = self.shift_array(list(target_quat))

                    break
            target_positions_interpolate.append(target_t)
            target_quaternions_interpolate.append(target_quat)
        return target_positions_interpolate, target_quaternions_interpolate


class KJRobotEnv:
    metadata = {'render.modes': ['human']}
    def __init__(self, cfg, render=True):
        super().__init__()
        # null space 设置
        self.lowerLimits = [-1.57, -1.57, -1.57, -1.57, -1.57, -1.57,
                            -1.57, -1.57, -1.57, -1.57, -1.57, -1.57]
        self.upperLimits = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57,
                            1.57, 1.57, 1.57, 1.57, 1.57, 1.57]
        self.jointRanges = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
                            3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
        self.restPoses = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.jointDamping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                             0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        """
        0-5 Right_Joint1-6
        6-11 Left_Joint1-6
        """
        # 实例化插值器
        self.interpolation = Interpolation()

        # 连接物理模拟
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self._timeStep = 120
        p.setTimeStep(1.0 / self._timeStep)

        # 加载机器人模型
        self.robotId = p.loadURDF("./assert/xy2leg3_4/urdf/xy2leg3_4.urdf",
                                  [0, 0, 0.7], useFixedBase=True)
        # 获取机器人可动关节信息
        movable_joints = get_movable_joints(self.robotId)
        self.lower_limits = [get_joint_limits(self.robotId, joint)[0] for joint in movable_joints]
        self.upper_limits = [get_joint_limits(self.robotId, joint)[1] for joint in movable_joints]
        joint_ranges = [upper - lower for lower, upper in zip(self.lower_limits, self.upper_limits)]
        self.current_conf = get_joint_positions(self.robotId, movable_joints)
        self.right_arm_joints = movable_joints[0:6]
        self.left_arm_joints = movable_joints[6:12]
        self.left_ee_link = 13
        self.right_ee_link = 6
        self.right_index = np.array([0, 1, 2, 3, 4, 5])
        self.left_index = np.array([6, 7, 8, 9, 10, 11])
        # 设置机器人起始位姿
        robotStartPos = [0, 0, 0.7]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.robotId, robotStartPos, robotStartOrientation)

        # 刷新环境
        p.setRealTimeSimulation(1)

    def calIKleft(self, target_pos, target_quat, resetPoses):
        """ 使用pybullet计算小机器人逆运动学 """
        # self.restPoses[6:] = resetPoses
        target_joint_value = p.calculateInverseKinematics(
            bodyUniqueId=self.robotId,
            endEffectorLinkIndex=11,
            targetPosition=target_pos,
            targetOrientation=target_quat,
            jointDamping=self.jointDamping,
            upperLimits=self.upperLimits,
            lowerLimits=self.lowerLimits,
            jointRanges=self.jointRanges,
            restPoses=self.restPoses,
            maxNumIterations=5000,
            residualThreshold=1e-1
        )

        # 在物理引擎中也同步运动到指定位置
        # for i in range(6):
        #     p.resetJointState(bodyUniqueId=self.robotId, jointIndex = i + 6, targetValue=target_joint_value[i+6])
        # p.stepSimulation()

        return target_joint_value

    def calIKright(self, target_pos, target_quat, resetPoses):
        # self.restPoses[0:6] = resetPoses
        """ 使用pybullet计算小机器人逆运动学 """
        target_joint_value = p.calculateInverseKinematics(
            bodyUniqueId=self.robotId,
            endEffectorLinkIndex=5,
            targetPosition=target_pos,
            targetOrientation=target_quat,
            jointDamping=self.jointDamping,
            upperLimits=self.upperLimits,
            lowerLimits=self.lowerLimits,
            jointRanges=self.jointRanges,
            restPoses=self.restPoses,
            maxNumIterations=5000,
            residualThreshold=1e-1
        )

        # 在物理引擎中也同步运动到指定位置
        # for i in range(6):
        #     p.resetJointState(bodyUniqueId=self.robotId, jointIndex = i, targetValue=target_joint_value[i])
        # p.stepSimulation()

        return target_joint_value

    def reset(self):
        return 1

    def run(self):
        # 手写轨迹
        # =====左手==========================
        # 支撑相
        # target_positions_left1 = np.array([
        #     [ 0.265, 0.05, -0.316],
        #     [ 0.088,  0.05, -0.316],
        #     [-0.088,  0.05, -0.316],
        #     [-0.265, 0.05, -0.316],
        # ])
        target_positions_left = np.array([
            [0.265, 0.05, -0.8],
            [0.1, 0.05, -0.8],
            # [ 0,  0, -0.7],
            # [0.05, 0, -0.7],
            # [-0.145, 0.05, -0.45],
            # [-0.165,  0.05, -0.45],
            # [-0.185,  0.05, -0.45],
            # [-0.215,  0.05, -0.45],
            # [-0.265, 0.05, -0.45],
        ])
        target_quaternions_left = np.array([
            [0.0, 0.0, 0.0, 1],
            [0.0, 0.0, 0.0, 1],
            # [0.0, 0.707, 0.0, 0.707],
            # [0.0, 0.0, 0.0, 1],
            # [0, 0, 0, 1],
            # [0, 0, 0, 1],
            # [0, 0, 0, 1],
            # [0.0, 0.0, 0.0, 1],
        ])
        target_positions_right = np.array([  # -0.05
            [0.265, -0.05, -0.8],
            [0.1, -0.05, -0.8],
            # [ 0.0,  0.0, -0.7],
            # [0.05, 0.0, -0.7],
            # [-0.145, -0.05, -0.45],
            # [-0.165,  -0.05, -0.45],
            # [-0.185,  -0.05, -0.45],
            # [-0.215,  -0.05, -0.45],
            # [-0.265, -0.05, -0.45],
        ])
        target_quaternions_right = np.array([
            [0.0, 0.0, 1, 0.0],
            [0.0, 0.0, 1, 0.0],
            # [0.707, 0.0, 0.707, 0.0],
            # [0.0, 0.0, 1, 0.0],
            # [0, 0, 1, 0],
            # [0, 0, 1, 0],
            # [0, 0, 1, 0],
            # [0.0, 0.0, 1, 0.0],
        ])

        # 测试插值之前路点
        i = 0
        init_joint_values_left = np.zeros((len(target_positions_left), 12))
        for t, r in zip(target_positions_left, target_quaternions_left):
            if i == 0:
                init_joint_values_left[i] = self.calIKleft(t, r, np.array([0, 0, 0, 0, 0, 0]))
            else:
                init_joint_values_left[i] = self.calIKleft(t, r, init_joint_values_left[i - 1])
            i += 1
        init_joint_values_left = init_joint_values_left[:, 6:]

        j = 0
        init_joint_values_right = np.zeros((len(target_positions_right), 12))
        for t, r in zip(target_positions_right, target_quaternions_right):
            if j == 0:
                init_joint_values_right[j] = self.calIKright(t, r, np.array([0, 0, 0, 0, 0, 0]))
            else:
                init_joint_values_right[j] = self.calIKright(t, r, init_joint_values_right[j - 1])
            j += 1
        init_joint_values_right = init_joint_values_right[:, 0:6]


        for r_joint, l_joint in zip(init_joint_values_right, init_joint_values_left):
            for i, j, k in zip(self.right_index, self.left_index, range(6)):
                p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=r_joint[k], force=500)
                p.setJointMotorControl2(self.robotId, j, p.POSITION_CONTROL, targetPosition=l_joint[k], force=500)

            self.wait_n_steps(10, self.physicsClient)


        # 插值
        total_ts = np.array([0, 0.001, 0.6])
        target_positions_left1, target_quaternions_left1 = self.interpolation.mutiPointInterpolation(target_positions_left,
                                                                                                target_quaternions_left,
                                                                                                total_ts)

        target_joint_values_left1 = np.zeros((len(target_positions_left1), 12))
        for i in range(len(target_positions_left1)):
            if i == 0:
                target_joint_values_left1[i] = self.calIKleft(target_positions_left1[i],
                                                                      target_quaternions_left1[i],
                                                                      np.array([0, 0, 0, 0, 0, 0]))
            else:
                target_joint_values_left1[i] = self.calIKleft(target_positions_left1[i],
                                                                      target_quaternions_left1[i],
                                                                      target_joint_values_left1[i - 1, 6:])
        target_joint_values_left = target_joint_values_left1[:, 6:]

        # =====右手==========================
        # 支撑相
        # target_positions_right1 = np.array([  # -0.05
        #     [ 0.265, -0.05, -0.316],
        #     [ 0.088,  -0.05, -0.316],
        #     [-0.088,  -0.05, -0.316],
        #     [-0.265, -0.05, -0.316],
        # ])

        # 插值
        total_ts = np.array([0, 0.001, 0.6])
        target_positions_right1, target_quaternions_right1 = self.interpolation.mutiPointInterpolation(
            target_positions_right, target_quaternions_right, total_ts)

        target_joint_values_right1 = np.zeros((len(target_positions_right1), 12))
        for i in range(len(target_positions_right1)):
            if i == 0:
                target_joint_values_right1[i] = self.calIKright(target_positions_right1[i],
                                                                        target_quaternions_right1[i],
                                                                        np.array([0, 0, 0, 0, 0, 0]))
            else:
                target_joint_values_right1[i] = self.calIKright(target_positions_right1[i],
                                                                        target_quaternions_right1[i],
                                                                        target_joint_values_right1[i - 1, 6:])
        target_joint_values_right = target_joint_values_right1[:, 0:6]
        # 保存关节角度到 txt 文件
        try:
            np.savetxt('right_joints.txt', target_joint_values_right, fmt='%.6f', delimiter=' ')
            print("Saved right joints to right_joints.txt")
        except Exception as e:
            print(f"Error saving right_joints.txt: {e}")

        try:
            np.savetxt('left_joints.txt', target_joint_values_left, fmt='%.6f', delimiter=' ')
            print("Saved left joints to left_joints.txt")
        except Exception as e:
            print(f"Error saving left_joints.txt: {e}")

        # 测试插值之后路点
        # print('ik_r',ik_sol_r)
        for r_joint, l_joint in zip(target_joint_values_right, target_joint_values_left):
            for i, j, k in zip(self.right_index, self.left_index, range(len((target_joint_values_right[i])))):
                p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=r_joint[k], force=500)
                p.setJointMotorControl2(self.robotId, j, p.POSITION_CONTROL, targetPosition=l_joint[k], force=500)

            self.wait_n_steps(10, self.physicsClient)


        return {
            'states': target_positions_right1
        }

    def step(self, action):
        dt = 1.0 / self._timeStep
        n_steps = self._timeStep // self.control_hz
        if action is not None:
            self.latest_action = action
            action_position = self.hole_up_end
            action_orientation_6d = np.array(action[0:self.action_dim])
            # action_orientation = _6d_to_quaternion(action_orientation_6d)
            for i in range(n_steps):
                # ------------------------------------------求解器-------------------------------------------------------

                self.target_joint = p.calculateInverseKinematics(
                    bodyUniqueId=self.ur5_id,
                    endEffectorLinkIndex=7,
                    targetPosition=list(action_position),
                    targetOrientation=list(action_orientation_6d),
                    jointDamping=[0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                    physicsClientId=self.physicsClient_use)

                # self.control_jointsArray_to_target(self.ur5_id, list(self.target_joint),
                #                                    [1,2,3,4,5,6], physicsClientId=self.physicsClient_use)
                self.control_joints_to_target(self.ur5_id, list(self.target_joint),
                                                   [1,2,3,4,5,6], physicsClientId=self.physicsClient_use)



            # 计算当前状态
            robot_state = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
            robot_srate_orientation = np.array(robot_state[1])
            end_orn_euler = R.from_quat(robot_state[1]).as_euler('xyz', degrees=True)
            robot_state_position = np.array(robot_state[0])
            FT = self.getForceTorque()
            current_force = np.array([FT[0], FT[1], FT[2]])
            print("---- force ---", current_force)

            if np.linalg.norm(current_force[2]) < -10:
                robot_state_position[2] -= 0.009

                self.target_joint = p.calculateInverseKinematics(
                    bodyUniqueId=self.ur5_id,
                    endEffectorLinkIndex=7,
                    targetPosition=list(robot_state_position),
                    targetOrientation=list(robot_srate_orientation),
                    jointDamping=[0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                    physicsClientId=self.physicsClient_use)
                self.control_joints_to_target(self.ur5_id, list(self.target_joint),
                                              [1, 2, 3, 4, 5, 6], physicsClientId=self.physicsClient_use)

        observation = self.get_observation()
        reward = 1
        done = False
        info = None

        return observation, reward, done, info

    def shift_array(self, arr):
        if len(arr) <= 1:
            return arr
        first_ele = arr.pop(0)
        arr.append(first_ele)

        return arr

    def setup_control_joint(self, robotID, ControlJoints):
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(robotID)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                "controllable"])
        self.joints = AttrDict()
        self.control_joint_ids = []
        for i in range(numJoints):
            info = p.getJointInfo(robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in ControlJoints else False
            self.controlJointsInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                                               jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            # if info.type == "REVOLUTE":  # set revolute joint to static
            #     p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[self.controlJointsInfo.name] = self.controlJointsInfo
            if controllable:
                self.control_joint_ids.append(self.controlJointsInfo[0])

        return self.joints

    def control_joints_to_target(self, robotID, jointPose, Jointindex, physicsClientId):
        j = 0
        for i in Jointindex:
            forcemaxforce = 500
            p.setJointMotorControl2(bodyUniqueId=robotID,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPose[j],
                                    targetVelocity=0.0,
                                    force=forcemaxforce,
                                    maxVelocity=1,
                                    positionGain=0.03,
                                    velocityGain=1,
                                    physicsClientId = physicsClientId)
            j = j+1
        self.wait_n_steps(240, physicsClientId)

    def getForceTorque(self):
        """
        获取机器人当前关节力/力矩
        """
        FT = []
        force = np.array(p.getJointState(self.ur5_id, 7)[2][0:3], dtype=float).reshape(3, 1)
        torque = np.array(p.getJointState(self.ur5_id, 7)[2][3:6], dtype=float).reshape(3, 1)
        FT = [force[0][0], force[1][0], force[2][0], torque[0][0], torque[1][0], torque[2][0]]
        return FT

    def go(self, target_pos, target_orie):
        # ------------------------------------------求解器-------------------------------------------------------
        self.target_robot_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_id,
            endEffectorLinkIndex=7,
            targetPosition=target_pos,
            targetOrientation=target_orie,
            jointDamping=self.joint_damping, )

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.target_robot_joint_angles),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(300):
            p.stepSimulation()

    def control_jointsArray_to_target(self, robotID, pose_array, Jointindex, physicsClientId):
        p.setJointMotorControlArray(robotID, Jointindex,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=pose_array,
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]),
                                    physicsClientId = physicsClientId
                                    )
        p.setTimeStep(1.0 / self._timeStep)
        self.wait_n_steps(240, physicsClientId)

    def wait_n_steps(self, n: int, physicsClientId):
        for i in range(n):
            p.stepSimulation(physicsClientId = physicsClientId)

    def getJointStates(self, robotID, control_joint_index):
        joint_states = p.getJointStates(robotID, control_joint_index)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def uniform_sampling(self, point_cloud, filter_depth):
        condition = point_cloud[:, 2] < filter_depth
        filtered_points = point_cloud[condition, :]
        indices = np.random.permutation(filtered_points.shape[0])[:self.num_points]
        sampled_points = filtered_points[indices, :]
        return sampled_points

    def cal_planner(self, t0, R0, t1, R1, time):
        position_parameter = LinePositionParameter(t0, t1)
        attitude_parameter = OneAttitudeParameter(R0, R1)
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        trajectory_planner = TrajectoryPlanner(trajectory_parameter)
        return trajectory_planner

    def get_observation(self):
        self.camera_Position = p.getLinkState(self.ur5_id, 8, computeForwardKinematics=1)[0]
        self.camera_Orientation = p.getLinkState(self.ur5_id, 8, computeForwardKinematics=1)[1]
        self.goalPosition1.update(self.camera_Position, self.camera_Orientation)
        self.cube_position = p.getBasePositionAndOrientation(self.tool_id[0])[0]
        self.cube_position = [-0.35, 0.0 ,0.35]
        self.camera_in_world = [-0.6, 0.2, 0.7]
        self.view_matrix = p.computeViewMatrix(
                                            cameraEyePosition = [self.camera_Position[0]+0.07,
                                                                self.camera_Position[1]+0.03,
                                                                self.camera_Position[2]+0.06],
                                            # cameraEyePosition=[self.camera_in_world[0],
                                            #                    self.camera_in_world[1],
                                            #                    self.camera_in_world[2]],
                                            cameraTargetPosition = [self.cube_position[0],
                                                                    self.cube_position[1],
                                                                    self.cube_position[2]],
                                            cameraUpVector = [0, 0, 1])
        self.view_matrix_hand = p.computeViewMatrix(
                                            cameraEyePosition = [self.camera_Position[0]-0.07,
                                                                self.camera_Position[1]-0.03,
                                                                self.camera_Position[2]+0.06],
                                            cameraTargetPosition = [self.cube_position[0],
                                                                    self.cube_position[1],
                                                                    self.cube_position[2]],
                                            cameraUpVector = [0, 0, 1])
        # print("camera_Orientation: ", camera_Orientation)
        # print("position: ", camera_Position)

        # world
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0.1083366177733302, -0.4868269875385862, 1.0],
        #                                                   distance = .26,
        #                                                   yaw = 0,
        #                                                   pitch = -55,
        #                                                   roll = 0,
        #                                                   upAxisIndex = 2)

        actions = np.zeros((self.action_dim))
        # intrinsics of the camera
        self.fov = 80
        aspect = float(self.image_width / self.image_height)
        near = 0.001
        far = 20.0
        self.camera_matrix = np.array([
            [self.image_height / (2.0 * np.tan(self.fov / 2.0)), 0.0, self.image_width / 2.0],
            [0.0, self.image_height / (2.0 * np.tan(self.fov / 2.0)), self.image_height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, near, far)

        images = p.getCameraImage(width = self.image_width,
                                    height = self.image_height,
                                    viewMatrix = self.view_matrix,
                                    projectionMatrix = self.proj_matrix,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        images_hand = p.getCameraImage(width = self.image_width,
                                    height = self.image_height,
                                    viewMatrix = self.view_matrix_hand,
                                    projectionMatrix = self.proj_matrix,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        assert (images[0] == self.image_width or images_hand[0] == self.image_width)
        assert (images[1] == self.image_height or images_hand[1] == self.image_height)
        rgb_image = np.reshape(images[2], (self.image_height, self.image_width, 4)) * 1. / 255.
        rgb_image_hand = np.reshape(images_hand[2], (self.image_height, self.image_width, 4)) * 1. / 255.
        rgb_image_3_channel = np.uint8(rgb_image[:, :, :3] * 255)
        rgb_image_hand_3_channel = np.uint8(rgb_image_hand[:, :, :3] * 255)

        rgb_image = rgb_image[:, :, :3].astype(np.float32)
        obs_image = np.uint8(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) * 255)
        gray_image = np.expand_dims(obs_image, axis=2)
        assert (rgb_image.shape == (self.image_height, self.image_width, 3))
        assert (gray_image.shape == (self.image_height, self.image_width, 1))

        depth_buffer = np.reshape(images[3], [self.image_height, self.image_width])
        depth_image = far * near / (far - (far - near) * depth_buffer)

        seg_image = np.reshape(images[4], [self.image_height, self.image_width]) * 1. / 255.

        depth_image_uint8 = np.uint8(depth_image * (1. / depth_image.max()) * 255.)

        depth_image_uint8 = np.expand_dims(depth_image_uint8, axis=2)
        assert (depth_image_uint8.shape == (self.image_height, self.image_width, 1))
        depth_image_3_channel = np.concatenate((depth_image_uint8, depth_image_uint8, depth_image_uint8), axis=2)
        assert (rgb_image_3_channel.shape == (self.image_height, self.image_width, 3) or rgb_image_hand_3_channel.shape == (self.image_height, self.image_width, 3))

        """ save images """
        # cv2.imwrite("./rgb_image_3_channel.jpg", rgb_image_3_channel)
        # cv2.imwrite("./rgb_image_hand_3_channel.jpg", rgb_image_hand_3_channel)
        # cv2.imwrite("./depth_image_uint8.jpg", depth_image_uint8)

        """ the observation can be changed to perform ablative studies"""

        point_cloud = np.zeros((self.image_height * self.image_width, 6))
        for h in range(self.image_height):
            for w in range(self.image_width):
                point_cloud[h * self.image_width + w, :3] = self.camera_matrix_inv @ np.array(
                    [w * 1.0, h * 1.0, 1.0]) * depth_image[h, w]
                point_cloud[h * self.image_width + w, 3:] = rgb_image_3_channel[h, w, :]
        sampled_points = self.uniform_sampling(point_cloud, 0.8)

        robot_position = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
        # peg_transition = self.Visualize_rotation_center(robot_position[4], robot_position[5], relative_offset=[0.055, 0, 0], UI=False)
        self.peg_position = np.array(robot_position[0])
        self.peg_orientation = np.array(robot_position[1])

        peg_orientation_6d = quaternion_to_6d(self.peg_orientation, order='xyzw')
        action_position = np.array(self.peg_position)
        action_orientation = np.array(peg_orientation_6d)
        # actions[0:3] = action_position
        # actions[3:self.action_dim] = action_orientation
        actions[0:self.action_dim] = self.peg_orientation
        force = np.array(self.getForceTorque()[:3])

        obs = {
            'agent_pos': actions,
            'image': rgb_image_3_channel,
            'image_hand': rgb_image_hand_3_channel,
            'depth': depth_image_uint8,
            'point_cloud': sampled_points,
            'force': force
        }

        return obs

    def distance_to_goal(self):
        res = np.zeros(3)
        robot_position = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
        peg_position = self.Visualize_rotation_center(robot_position[4], robot_position[5], relative_offset=[0.055, 0, 0], UI=False)[0]
        hole_position = p.getBasePositionAndOrientation(self.tool_id[0])[0]

        robot_peg = np.array(peg_position)
        hole = np.array(hole_position)

        hole[2] = hole[2]- 0.08 # 孔深度0.08
        res[0] = np.abs(robot_peg[0] - hole[0])
        res[1] = np.abs(robot_peg[1] - hole[1])
        res[2] = np.linalg.norm(robot_peg - hole)
        return res

    def distance_to_goal_rotation(self):
        quat_rot_err = np.zeros(4)
        orientation_err = np.zeros(4)

        robot_orientation = np.array(p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex)[5])
        hole_orientation = Rotation.from_euler('xyz', [90, 90, -90], degrees=True).as_quat() # 默认固定孔的姿态

        # if np.dot(np.transpose(hole_orientation), robot_orientation) < 0.0:
        #     robot_orientation = -robot_orientation

        hole_orientation = Rotation.from_euler('xyz', [90, 90, -90], degrees=True).as_matrix() # 默认固定孔的姿态

        current_orie_matrix = Rotation.from_quat(robot_orientation).as_matrix()
        target_orie_inv = hole_orientation.T
        quat_rot_err_tmp = np.dot(current_orie_matrix, target_orie_inv)
        
        # ------ 轴角表示误差，不准 ------
        # quat_rot_err_tmp = Rotation.from_matrix(quat_rot_err_tmp).as_quat()
        # np.copyto(quat_rot_err, quat_rot_err_tmp)

        # if np.linalg.norm(quat_rot_err) > 1e-3:
        #     quat_rot_err = quat_rot_err / np.linalg.norm(quat_rot_err)
        # axis, angle = tfs.quaternions.quat2axangle(quat_rot_err) # 创建一个四元数
        # rotation_err = axis * angle
        # ------ 轴角表示误差，不准 ------

        rotation_err = Rotation.from_matrix(quat_rot_err_tmp).as_rotvec()

        orientation_err[0] = rotation_err[0]
        orientation_err[1] = rotation_err[1]
        # 将角度归一化到 [0, 2π) 范围内
        orientation_err[2] = rotation_err[2]

        return np.linalg.norm(orientation_err)

    def Visualize_rotation_center(self, end_pos, end_orn, relative_offset=[0.055, 0, 0], UI=True):
        """
           目的：可视化固定点旋转,并返回固定点位姿
           Arguments:
           - end_pos: len=3, 该 link 的在世界坐标系的位置
           - end_orn: len=4, 该 link 的在世界坐标系的姿态 (x, y, z, w)
           - relative_offset 该 link下的相对移动 list of 
           - relative_euler  该 link下的旋转  list of 3

           Returns:
           - [peg_link_pos,peg_link_Quaternion]
           """
        # 将eelink变换到peg末端，可视化坐标系
        peg_link_Rotation_matrix = relative_pos_and_ore_form_world(end_pos, end_orn,
                                                                   [relative_offset[0], relative_offset[1],
                                                                    relative_offset[2]], [0, 0, 0])
        peg_link_Quaternion = Rotation.from_matrix(peg_link_Rotation_matrix[:3, :3]).as_quat()
        peg_link_pos = peg_link_Rotation_matrix[:3, -1]
        if UI == True:
            self.Visualize_rotation_center_UI.update(peg_link_pos, peg_link_Quaternion)

        return [peg_link_pos, peg_link_Quaternion]

    def impedance_controller(self, action, physicsClientId):
        """
        z方向使用力控制
        x,y方向使用导纳控制
        input:
        out:
            z方向恒力控制
            x,y方向柔顺
        """
        action_to_target_pose = [None] * 3
        action_to_target_orie = [None] * 4
        action_to_current = [None] * 3
        desired_force_z = 0.1  # N
        desired_force_xy = 0.0  # N
        desired_force_rz = 0.0
        Kp_force = 5  # 比例增益
        Ki_force = 0.4  # 积分增益
        Kd_force = 0.01  # 微分增益
        integral_force_error_z = 0.0
        previous_force_error_z = 0.0
        delta_z_max = 2
        # 获取当前的关节力和力矩
        FT = self.getForceTorque()
        Fext = [FT[0], FT[1], FT[2]]
        Text = [FT[3], FT[4], FT[5]]
        # Text = [0, 0, 0]

        action_to_target_pose[0] = action[0] + self.zero_Position[0]
        action_to_target_pose[1] = action[1] + self.zero_Position[1]
        action_to_target_pose[2] = action[2] + self.zero_Position[2]

        action_to_target_orie[0] = action[3]
        action_to_target_orie[1] = action[4]
        action_to_target_orie[2] = action[5]
        action_to_target_orie[3] = action[6]

        wrench_z = Fext[2]
        # print("wrench_z : ", wrench_z)

        self.force_error_z = wrench_z - desired_force_z
        self.force_y = Fext[1]

        integral_force_error_z += self.force_error_z * self.duration

        # 计算微分项
        derivative_force_error_z = (self.force_error_z - previous_force_error_z) / self.duration
        previous_force_error_z = self.force_error_z

        # 计算力控制输出
        force_control_output_z = Kp_force * self.force_error_z + \
                                 Ki_force * integral_force_error_z + \
                                 0
        #  Kd_force * derivative_force_error_z
        force_z_adjustment = force_control_output_z
        pos_z_adjustment = max(min(force_z_adjustment, delta_z_max), -delta_z_max)
        pos_z_adjustment = pos_z_adjustment / 100  # 手动补偿
        if wrench_z > 0.1:
            print("wrench_z : ", wrench_z)
            pos_z_adjustment = pos_z_adjustment
        for i in range(1):
            if Fext[1] > 0.5 or Text[2] > 0.001 :
                desired_force_rz = -0.1
                desired_force_xy = -0.1
            force_external = np.mat([[Fext[0] - desired_force_xy], [Fext[1] - desired_force_xy], [0.0], [Text[0] - desired_force_rz], [Text[1] - desired_force_rz], [Text[2] - desired_force_rz]])
            # force_external = np.mat([[0], [Fext[1]], [0], [0], [0], [0]])

            # 获取当前的关节状态
            self.current_Position = p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[4]
            self.current_Orientation = np.array(p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[5])
            action_to_current[0] = self.current_Position[0] - action_to_target_pose[0]  # 计算当前位置与目标位置的差值
            action_to_current[1] = self.current_Position[1] - action_to_target_pose[1]
            action_to_current[2] = self.current_Position[2] - action_to_target_pose[2]

            self.orientation_err = np.dot(self.current_Orientation, self.zero_Orientation)
            orien_angle_err = 2 * np.arccos(self.orientation_err)
            self.angle_err = orien_angle_err * 180 / np.pi
            res = self.quaternion_to_euler(self.zero_Orientation, self.current_Orientation)
            # print("res: ", res)

            current_orie_matrix = Rotation.from_quat(self.current_Orientation).as_matrix()
            action_to_target_orie_matrix = Rotation.from_quat(self.zero_Orientation).as_matrix()
            target_orie_inv = action_to_target_orie_matrix.T
            quat_rot_err_tmp = np.dot(current_orie_matrix, target_orie_inv)

            quat_rot_err_ = Rotation.from_matrix(quat_rot_err_tmp).as_rotvec()
            quat_rot_err_tmp = quat_rot_err_ * 200
            # quat_rot_err_tmp = quat_rot_err_tmp * 180 / np.pi
            # print("quat_rot_err_tmp: ", quat_rot_err_tmp)

            # Position error
            self.dx = action_to_current[0] / 1.0
            self.dy = action_to_current[1] / 1.0
            self.dz = action_to_current[2] / 1.0
            self.rx = quat_rot_err_tmp[0]
            self.ry = quat_rot_err_tmp[1]
            self.rz = quat_rot_err_tmp[2]
            if self.rz < 1e-5:
                dx = 0
            # self.pose_err = np.mat([[0.0], [-self.dy], [0.0], [0.0], [0.0], [0.0]])
            self.pose_err = np.mat([[-self.dx], [-self.dy], [0.0], [-self.rx], [-self.ry], [-self.rz]])
            self.FT_err = np.mat([[0.0], [0.0], [pos_z_adjustment], [0.0], [0.0], [0.0]])

            coupling_wrench_arm = self.Inverse_M * force_external + self.stiffness * self.pose_err
            arm_desired_accelaration = -self.damping * self.arm_desired_twist_ + coupling_wrench_arm + self.FT_err

            arm_acc_norm = np.linalg.norm(arm_desired_accelaration[:, :3])
            if (arm_acc_norm > self.arm_max_acc_):
                # print("Admittance generates high arm accelaration!", arm_acc_norm)
                arm_desired_accelaration[:, :3] *= (self.arm_max_acc_ / arm_acc_norm)
            # print("pose_err : ", self.pose_err, self.FT_err)

            # 更新 arm_desired_twist_adm_
            deta_arm_desired_twist = arm_desired_accelaration * self.duration
            self.arm_desired_twist_ += deta_arm_desired_twist

        return self.arm_desired_twist_, deta_arm_desired_twist

    # def impedance_controller(self, action):
    #     """
    #     z方向使用力控制
    #     x,y方向使用导纳控制
    #     input:
    #     out:
    #         z方向恒力控制
    #         x,y方向柔顺
    #     """
    #     action_to_target_pose = [None] * 3
    #     action_to_target_orie = [None] * 4
    #     action_to_current = [None] * 3
    #     desired_force_z = 1  # N
    #     desired_force_xy = 0.0  # N
    #     desired_torque_x = 0.0
    #     desired_torque_y = 0.0
    #     desired_torque_z = 0.0
    #
    #     Kp_force = 5  # 比例增益
    #     Ki_force = 0.4  # 积分增益
    #     Kd_force = 0.01  # 微分增益
    #     integral_force_error_z = 0.0
    #     previous_force_error_z = 0.0
    #     delta_z_max = 1
    #     # 获取当前的关节力和力矩
    #     FT = self.getForceTorque()
    #     Fext = [FT[0], FT[1], FT[2]]
    #     Text = [FT[3], FT[4], FT[5]]
    #     # Text = [0, 0, 0]
    #
    #     action_to_target_pose[0] = action[0] + self.zero_Position[0]
    #     action_to_target_pose[1] = action[1] + self.zero_Position[1]
    #     action_to_target_pose[2] = action[2] + self.zero_Position[2]
    #
    #     action_to_target_orie[0] = action[3]
    #     action_to_target_orie[1] = action[4]
    #     action_to_target_orie[2] = action[5]
    #     action_to_target_orie[3] = action[6]
    #
    #     wrench_z = Fext[2]
    #     print("wrench_z : ", wrench_z)
    #
    #     self.force_error_z = wrench_z - desired_force_z
    #     self.force_y = Fext[1]
    #
    #     integral_force_error_z += self.force_error_z * self.duration
    #
    #     # 计算微分项
    #     derivative_force_error_z = (self.force_error_z - previous_force_error_z) / self.duration
    #     previous_force_error_z = self.force_error_z
    #
    #     # 计算力控制输出
    #     force_control_output_z = Kp_force * self.force_error_z + \
    #                              Ki_force * integral_force_error_z + \
    #                              0
    #     #  Kd_force * derivative_force_error_z
    #     force_z_adjustment = force_control_output_z
    #     pos_z_adjustment = max(min(force_z_adjustment, delta_z_max), -delta_z_max)
    #     pos_z_adjustment = pos_z_adjustment / 100  # 手动补偿
    #     if wrench_z > 0.1:
    #         print("wrench_z : ", wrench_z)
    #         pos_z_adjustment = pos_z_adjustment
    #     for i in range(1):
    #         if Fext[1] > 0.5:
    #             desired_force_xy = 0.5
    #         if Text[0] > 0.5:
    #             desired_torque_x = 0.005
    #         if Text[1] > 0.5:
    #             desired_torque_y = 0.005
    #         if Text[2] > 0.5:
    #             desired_torque_z = 0.005
    #         if np.abs(Fext[0]) > 1 or np.abs(Fext[1]) > 1 or np.abs(Fext[2]) > 1 or np.abs(Text[0]) > 1 or np.abs(Text[1]) > 1 or np.abs(Text[2]) > 1:
    #         # if 0:
    #             force_external = np.mat([[Fext[0]  - desired_force_xy], [Fext[1]  - desired_force_xy], [0], [0], [0], [0]])
    #         else:
    #             # force_external = np.mat([[Fext[0]  - desired_force_xy], [Fext[1] - desired_force_xy], [0], [Text[0] - desired_torque_x], [Text[1] - desired_torque_y], [Text[2] - desired_torque_z]])
    #             force_external = np.mat([[Fext[0]  - desired_force_xy], [Fext[1] - desired_force_xy], [0.0], [0.0], [0.0], [Text[2]]])
    #
    #         # 获取当前的关节状态
    #         self.current_Position = p.getLinkState(self.ur5_id, 7)[4]
    #         self.current_Orientation = np.array(p.getLinkState(self.ur5_id, 7)[5])
    #         action_to_current[0] = self.current_Position[0] - action_to_target_pose[0]  # 计算当前位置与目标位置的差值
    #         action_to_current[1] = self.current_Position[1] - action_to_target_pose[1]
    #         action_to_current[2] = self.current_Position[2] - action_to_target_pose[2]
    #
    #         orientation_err = np.dot(self.current_Orientation, self.zero_Orientation)
    #         orien_angle_err = 2 * np.arccos(orientation_err)
    #         angle_err = orien_angle_err * 180 / np.pi
    #         res = self.quaternion_to_euler(self.zero_Orientation, self.current_Orientation)
    #         print("res: ", res)
    #
    #         current_orie_matrix = Rotation.from_quat(self.current_Orientation).as_matrix()
    #         action_to_target_orie_matrix = Rotation.from_quat(self.zero_Orientation).as_matrix()
    #         target_orie_inv = action_to_target_orie_matrix.T
    #         quat_rot_err_tmp = np.dot(current_orie_matrix, target_orie_inv)
    #
    #         quat_rot_err_tmp = Rotation.from_matrix(quat_rot_err_tmp).as_rotvec()
    #         # quat_rot_err_tmp = quat_rot_err_tmp * 180 / np.pi
    #         print("quat_rot_err_tmp: ", quat_rot_err_tmp)
    #
    #         # Position error
    #         self.dx = action_to_current[0] / 1.0
    #         self.dy = action_to_current[1] / 1.0
    #         self.dz = action_to_current[2] / 1.0
    #         self.rx = quat_rot_err_tmp[0]
    #         self.ry = quat_rot_err_tmp[1]
    #         self.rz = quat_rot_err_tmp[2]
    #         # if dx < 1e-4:
    #         #     dx = 0
    #         if np.abs(Fext[0]) > 1 or np.abs(Fext[1]) > 1 or np.abs(Fext[2]) > 1 or np.abs(Text[0]) > 1 or np.abs(Text[1]) > 1 or np.abs(Text[2]) > 1:
    #         # if 0:
    #             self.pose_err = np.mat([[-self.dx], [-self.dy], [0.0], [0.0], [0.0], [0.0]])
    #             self.arm_desired_twist_[3:6,:] = 0.0
    #             self.arm_desired_twist_[0:2,:] = 0.0
    #             pos_z_adjustment = 0.05
    #             self.dx = 0
    #             self.dy = 0
    #         else:
    #             # self.pose_err = np.mat([[-self.dx], [-self.dy], [0.0], [self.rx], [self.ry], [self.rz]])
    #             self.pose_err = np.mat([[-self.dx], [-self.dy], [0.0], [0.0], [0.0], [self.rz]])
    #         self.FT_err = np.mat([[0.0], [0.0], [pos_z_adjustment], [0.0], [0.0], [0.0]]) # 端面不平
    #
    #         self.arm_desired_twist_[np.abs(self.arm_desired_twist_) < 1e-7] = 0.0
    #         self.pose_err[np.abs(self.pose_err) < 1e-7] = 0.0
    #         coupling_wrench_arm = self.Inverse_M * force_external + self.stiffness * self.pose_err
    #         arm_desired_accelaration = -self.damping * self.arm_desired_twist_ + coupling_wrench_arm + self.FT_err
    #
    #         arm_acc_norm = np.linalg.norm(arm_desired_accelaration[:, :3])
    #         if (arm_acc_norm > self.arm_max_acc_):
    #             # print("Admittance generates high arm accelaration!", arm_acc_norm)
    #             arm_desired_accelaration[:, :3] *= (self.arm_max_acc_ / arm_acc_norm)
    #         print("pose_err : ", self.pose_err, self.FT_err)
    #
    #         # 更新 arm_desired_twist_adm_
    #         deta_arm_desired_twist = arm_desired_accelaration * self.duration
    #         self.arm_desired_twist_ += deta_arm_desired_twist
    #
    #     return self.arm_desired_twist_, deta_arm_desired_twist

    def apply_hybrid_controller(self, action, physicsClientId):
        """ Make a step in simulation """
        position_arrive = False
        """
        k=40,d=28.28,m=10
        """
        self.translational_stiffness = 40
        self.rotational_stiffness = 40
        self.translational_damping = 28.28
        self.translational_damping = 28.28
        self.duration = 0.001
        self.zero_Orientation[0] = action[3]
        self.zero_Orientation[1] = action[4]
        self.zero_Orientation[2] = action[5]
        self.zero_Orientation[3] = action[6]
        self.reset_controller()
        while 1:
            desired_twist, deta_desired_twist = self.impedance_controller(action, physicsClientId)
            desired_twist = np.array(desired_twist)
            self.send_commands_to_robot(desired_twist[0], desired_twist[1], desired_twist[2], desired_twist[3], desired_twist[4], desired_twist[5], physicsClientId)
            self.solve_steps = self.solve_steps + 1
            # print("self.force_error_z : ", self.force_error_z)
            joint_positions = p.getJointState(self.ur5_id, 6, physicsClientId=physicsClientId)[0]
            # print("self.position_error_y : ", self.position_error_y)
            self.writer.add_scalars("force_error_z",
                                   {"force_error_z": self.force_error_z}, self.solve_steps)
            self.writer.add_scalars("rz",
                                   {"rz": self.rz}, self.solve_steps)
            self.writer.add_scalars("force_y",
                                   {"force_y": self.force_y}, self.solve_steps)
            self.writer.add_scalars("joint_positions",
                                   {"joint_positions": joint_positions}, self.solve_steps)
            if self.rz < 5e-6 or self.is_in_range(1, self.orientation_err, 5e-6) or self.angle_err < 0.1 or self.angle_err==None:
                # print("force err success")
                break

    def quaternion_to_euler(self, q1, q2):
        # 将四元数转换为欧拉角(ZYX顺序)
        rot1 = Rotation.from_quat(q1)
        rot2 = Rotation.from_quat(q2)
        euler1 = rot1.as_euler('zyx', degrees=True)
        euler2 = rot2.as_euler('zyx', degrees=True)

        # 计算角度差异(处理角度环绕)
        diff = np.abs(euler2 - euler1)
        diff = np.where(diff > 180, 360 - diff, diff)

        return diff

    def is_in_range(self, x, number, tol):
        return x-tol <= number <= x+tol

    def send_commands_to_robot(self, vx, vy, vz, wx, wy, wz, physicsClientId):
        # 获取当前末端执行器的位置和姿态
        end_pos, end_ori, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_ve = p.getLinkState(self.ur5_id,
                                                                                                    self.ur5EndEffectorIndex,
                                                                                                    computeLinkVelocity=1,
                                                                                                    computeForwardKinematics=1,
                                                                                                    physicsClientId=physicsClientId)
        joint_indices = []
        num_joints = p.getNumJoints(self.ur5_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.ur5_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # 过滤掉固定关节
                joint_indices.append(i)
        # 获取当前关节状态
        num_joints = len(joint_indices)
        joint_positions = [p.getJointState(self.ur5_id, i, physicsClientId=physicsClientId)[0] for i in joint_indices ]
        joint_velocities = [0] * num_joints  # 假设初始关节速度为0


        # 计算雅可比矩阵
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.ur5_id, self.ur5EndEffectorIndex, localPosition=com_trn,
                                                                objPositions=joint_positions,
                                                                objVelocities=joint_velocities,
                                                                objAccelerations=[0] * num_joints,
                                                                physicsClientId=physicsClientId
                                                                )
        # 将线性和角速度的雅可比矩阵合并
        jacobian = np.vstack((linear_jacobian, angular_jacobian))

        # 笛卡尔空间的线速度和角速度
        cartesian_velocity = np.array([vx, vy, vz, wx, wy, wz])  # vx, vy, vz 是线速度，wx, wy, wz 是角速度

        # 计算关节速度
        joint_velocities = np.linalg.pinv(jacobian) @ cartesian_velocity
        for i, joint_index in enumerate(joint_indices):
            p.setJointMotorControl2(self.ur5_id, joint_index, p.VELOCITY_CONTROL, targetVelocity=joint_velocities[i], force=500, physicsClientId=physicsClientId)
                # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(240):
            p.stepSimulation()