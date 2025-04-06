#!/usr/bin/env python
# -*- coding: utf-8 -*-

import skfuzzy as fuzz
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


class UR5Env:
    metadata = {'render.modes': ['human']}
    def __init__(self, render=True):
        super().__init__()
        self.log =[]

        self.randm_num = 1
        self.writer = SummaryWriter('./paperforceslog2')
        self.mointor_force_torque = np.zeros((2, 60))
        self.neibu = False
        self.sucessful_number = 0 # 迭代成功的次数
        self.all_number = 0  # 迭代次数
        self.epsiode_timesteps = 0
        self.max_steps_one_episode = 50#26
        self.step_counter = 0
        self.goal_cont = 0

        self.quat_rot_err = np.zeros(4)
        self.current_twist_lin = np.zeros((3, 1))
        self.current_twist_ang = np.zeros((3, 1))
        self.arm_desired_twist_ = np.mat(np.zeros((6, 1)))
        self.arm_desired_position_ = np.mat(np.zeros((3, 1)))
        self.arm_max_acc_ = 1
        self.duration = 0.2
        self.obj_t = np.zeros(3)
        self.num_points = 4096 * 2
        self.control_hz = 100

        self.Visualize_rotation_center_UI = DebugAxes()  # 可视化旋转中心
        self.goalPosition1 = DebugAxes()  # 可视化 eelink 坐标
        self.goalPosition_eelink = DebugAxes()  # 可视化 eelink 坐标

        self.image_width = 320
        self.image_height = 240

        # 机械臂实际执行频率，ur5真实通讯频率是120hz
        self._timeStep = 100
        self.t = (1 / self._timeStep)*2
        self.T0 = sm.SE3()

        self.robot_control_joint_name = ["shoulder_pan_joint",
                                         "shoulder_lift_joint",
                                         "elbow_joint",
                                         "wrist_1_joint",
                                         "wrist_2_joint",
                                         "wrist_3_joint"]

        # --------------------------- 重置关节至初始状态--------------------------------

        # self.init_joint_val =[0.17195291679571792, -1.2151152721500111, -2.042115573329094,
        #                       -1.4551581329978485, 1.5707963241241731, 0.17195291700664328]#标准垂直姿态
        # self.init_joint_val =[0.10195291679571792, -1.2151152721500111, -2.050115573329094,
        #                       -1.4551581329978485, 1.5707963241241731, 0.17195291700664328]#标准垂直姿态,接触桌面
        self.init_joint_val =[-0.50195291679571792, -1.2151152721500111, -2.042115573329094, # 0.10195291679571792
                              -1.4551581329978485, 1.5707963241241731, 0.17195291700664328]#标准垂直姿态,不接触桌面
        # 插孔是否失败阈值
        self.ftmax = [200, 100]
        self.threshold = [150, 100]

        # 定义初始值
        self.done = False
        self.tool_pos = [-0.3992177129905079, 0.04993405718861182, 0.4478437960506524]

        self.deuler = [0, 0, 0]
        self.dpos = [0, 0, 0]

        self.euler_init = [2.50928407, 1.22967306, -0.59999855]
        # 目标的插孔深度
        self.depth = 0.05
        # ---------------------------------定义初始状态--------------------------------------------
        self.current_pos = []
        self.current_orie = []
        self.state = 0
        self.next_state = 0
        # 定义初始动作
        # self.action = 0
        # 初始姿态
        self.init_joint_positions = [0, -1.203, -1.799, -1.69, 1.57, 0]
        # 初始末端工具的四元数
        self.init_orie_tool = [0.7070727237014777, 0.0, 0.0, 0.7071408370313327]
        # 机器人初始四元数
        self.init_orie = [-0.6997116804122925, -0.0003609205596148968, 0.7144252061843872, 0.0002908592578023672]
        self.joint_indices = [0,1,2,3,4,5]  # 你的机器人关节索引列表
        # --------------------------------------------------------------------------

        # 关节跳跃值
        self.joint_damping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

        # 自带数据库地址
        self.urdf_root_path = pybullet_data.getDataPath()

        ##  第一步，连接仿真环境
        self.is_render = render
        if self.is_render:
            self.physicsClient_use = p.connect(p.GUI)
            self.physicsClient_plan = p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)
        # 设定界面显示视角
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # -----------------------------------------------------------------------添加模型-----------------------------------------------------------------------------------------------
        # 添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.tool_id = p.loadSDF( "/home/zhou/autolab/peg-in-hole/888/file_recv/model/fangkuai/model.sdf")
        """ 用于测试恒力跟踪"""
        p.changeDynamics(self.tool_id[0], -1,
                         lateralFriction=100, spinningFriction=100, rollingFriction=0, frictionAnchor=True)

        #  直的
        p.resetBasePositionAndOrientation(self.tool_id[0], [-0.4 + 0.05, 0.1 - 0.05, 0.32 + 0.08],
                                          p.getQuaternionFromEuler([(3.145926 / 2), 0, 0]))#100宽度*100高*孔21
        self.init_height = 0.32+0.08
        self.goalPosition =[-0.4+0.05, 0.1-0.05, 0.32+0.08]

        # 添加桌子模型
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[-1.01, 0, -0.315])

        # 添加机器人模型
        self.ur5_id = p.loadURDF(
            "./assert/ur_description/urdf/ur5_robot_sensor_pos2_stand_eelink.urdf",
            basePosition=[0, 0, 0.1], flags=9)
        self.ur5_id_plan = p.loadURDF(
            "./assert/ur_description/urdf/ur5_robot_sensor_pos2_stand_eelink.urdf",
            basePosition=[0, 0, 0.1], flags=9, physicsClientId = self.physicsClient_plan)
        self.ur5EndEffectorIndex = 7
        self.numdof = 6
        self.numjoint = p.getNumJoints(self.ur5_id)
        for j in range(p.getNumJoints(self.ur5_id)):
            print(j, p.getJointInfo(self.ur5_id, j))
        link_color = [[1,0,0,1], # red
                      [0,1,0,1], # green
                      [0,0,1,1]] # blue

        # p.changeVisualShape(self.ur5_id, 9, rgbaColor=link_color[2])
        hole_position = p.getBasePositionAndOrientation(self.tool_id[0])[0]
        hole_orientation = p.getBasePositionAndOrientation(self.tool_id[0])[1]
        self.obj_t = hole_position
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # obs = self.get_observation()
        # observation_vw_shape = obs["achieved_goal"].shape
        # observation_depth_shape = obs["desired_goal"].shape

        # # 状态空间的值后续再研究
        # self.observation_space = gym.spaces.Dict(
        #     dict(
        #         achieved_goal=gym.spaces.Box(-100.0, 100.0, shape=observation_vw_shape, dtype=np.float32),
        #         desired_goal=gym.spaces.Box(-100.0, 100.0, shape=observation_depth_shape, dtype=np.float32),
        #     )
        # )
        # ------------------------------| modified by YbZhou |--------------------------
        self.position_x_low     = -0.1
        self.position_x_high    = 0.1
        self.position_y_low     = -0.1
        self.position_y_high    = 0.1
        self.position_z_low     = -0.1
        self.position_z_high    = 0
        self.orientation_x_low  = -1
        self.orientation_x_high = 1
        self.orientation_y_low  = -1
        self.orientation_y_high = 1
        self.orientation_z_low  = -1
        self.orientation_z_high = 1
        self.orientation_w_low  = -1
        self.orientation_w_high = 1

        self.pose_adjust_low     = 0 
        self.pose_adjust_high    = 1
        self.orien_adjust_low    = 0
        self.orien_adjust_high   = 1
        # 构建导纳控制的三个系数矩阵
        """
        when k=80, d=40, m=10 => w=5 l=1 : need 13 steps
        when k=80, d=56, m=10 => w=2.82 l=1 : need 17 steps
        """
        self.In_M = 0.1 # M = 10， inverse_M = 1/M 
        self.translational_stiffness = 80
        self.rotational_stiffness = 10
        self.translational_damping = 40
        self.translational_damping = 10

        self.Inverse_M = np.mat(self.In_M * np.eye(6))

        self.stiffness = np.mat(np.block([
            [self.translational_stiffness * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), self.rotational_stiffness * np.eye(3)]
        ]))

        self.damping = np.mat(np.block([
            [self.translational_damping * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), self.translational_damping * np.eye(3)]
        ]))

        # 设定参数动作空间
        self.action_space = spaces.Box(low=np.array([self.position_x_low,self.position_y_low,self.position_z_low,
                                                     self.orientation_x_low,self.orientation_y_low,self.orientation_z_low,self.orientation_w_low,
                                                     self.pose_adjust_low, self.orien_adjust_low]),
                                       high=np.array([self.position_x_high,self.position_y_high,self.position_z_high,
                                                      self.orientation_x_high,self.orientation_y_high,self.orientation_z_high,self.orientation_w_high,
                                                      self.pose_adjust_high, self.orien_adjust_high]), 
                                       dtype=np.float32)
        # ------------------------------| modified by YbZhou |--------------------------
        # 设置Z方向的重力
        p.setGravity(0, 0, 0)

    def reset(self):
        self.goal_cont=0
        self.randm_num += 1
        self.int_falg = True
        self.mointor_force_torque = np.zeros((2, 60))
        p.enableJointForceTorqueSensor(self.ur5_id, 7)
        p.stepSimulation()

        # --------------------------------------- 重置关节至初始状态------------------------------------这里有坑，p.resetJointState与p.setTimeStep()会导致初始姿态偏移
        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.init_joint_val[i])
            p.resetJointState(bodyUniqueId=self.ur5_id_plan, jointIndex=i + 1, targetValue=self.init_joint_val[i], physicsClientId=self.physicsClient_plan)

        self.zero_Position = np.zeros(3)
        self.zero_Orientation = np.zeros(4)

        # 定义平移
        current_pos = p.getLinkState(self.ur5_id, 7)[4]
        current_orie = p.getLinkState(self.ur5_id, 7)[5]
        #打印出效果用
        self.inint_orie_print =current_orie
        self.goalPosition_eelink.update(current_pos,current_orie)
        random.seed(self.randm_num)
        theta = random.uniform(0, 2*math.pi)
        deltaR = random.uniform(0.0000, 0.0003)
        init_dpos_noise_base = [0.00, deltaR*math.cos(theta), deltaR*math.sin(theta)]
        # init_dpos_noise_base = [0.0, 0.0,  0.07]

        current_pos1 = [current_pos[0] - init_dpos_noise_base[2], current_pos[1] - init_dpos_noise_base[1],
                     current_pos[2] + init_dpos_noise_base[0]]

        # 产生一个随机初始位姿(变欧拉角)
        random.seed(self.randm_num)
        init_euler_end_y_z = [0, (-1) ** (random.randrange(1, 3)) * random.uniform(0.0, 0.035),
                              (-1) ** (random.randrange(1, 3)) * random.uniform(0.0, 0.035)]

        # init_euler_end_y_z=[0,0.0,0.0]

        # 最原始的绝对坐标系下初始化
        eeink_link_next_Rotation_matrix = fix_center_rotation(current_pos1, current_orie,
                                                              [0, 0, 0], init_euler_end_y_z)

        init_Quaternion = Rotation.from_matrix(eeink_link_next_Rotation_matrix[:3, :3]).as_quat()
        # init_dpos_noise = eeink_link_next_Rotation_matrix[:3, -1]
        # self.init_move_robot(init_dpos_noise, init_Quaternion)

        # '''打印用向量计算X轴角度差异'''
        # error_angle = self.calculate_angle([1, 0, 0], np.array(
        #     np.linalg.inv(Rotation.from_quat(current_orie).as_matrix())
        #     @ np.array(p.getMatrixFromQuaternion(init_Quaternion)).reshape(3, 3) @ [1, 0, 0]).reshape(-1))  # 标准方程,单个向量求解

        # 初始化姿态(执行）
        # self.init_move_robot(init_dpos_noise,init_Quaternion)
        self.old_euler = np.array(p.getEulerFromQuaternion(init_Quaternion))
        self.initial_depth = 0.08
        self.initial_distance = self.distance_to_goal()[2]
        self.initial_distance_rotation = self.distance_to_goal_rotation()
        self.max_step_count = 500

        # 计算当前深度
        po = p.getLinkState(self.ur5_id, 7)
        end_ = self.Visualize_rotation_center(po[4], po[5], relative_offset=[0.055, 0, 0], UI=True)[0]

        # now_distance = np.sqrt(np.sum(np.square(end_[0:2] - self.goalPosition[0:2])))
        # print("初始化距离： ",now_distance)

        h = self.init_height - end_[2]  # 当前插入深度
        self.old_h = h

        del  init_Quaternion

        self.step_counter = 0
        self.force_reward =0
        # 返回初始的观测量
        return self.get_observation()

    def run(self):
        time0 = 0.001
        robot_position = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
        # peg_transition = self.Visualize_rotation_center(robot_position[4], robot_position[5], relative_offset=[0.055, 0, 0], UI=False)
        peg_position = robot_position[0]
        peg_orientation = robot_position[1]

        end_peg_matrix = np.eye(4)
        end_peg_matrix[:3, :3] = R.from_quat(peg_orientation).as_matrix()
        end_peg_matrix[:3, 3] = peg_position

        T0 = SE3(end_peg_matrix)
        t0 = T0.t
        R0 = sm.SO3(T0.R)
        t1 = t0.copy()
        R1 = R0.copy()
        planner0 = self.cal_planner(t0, R0, t1, R1, time0)

        time1 = 4.0
        t2 = t1.copy()
        self.hole_rt_end = np.zeros(3)
        self.hole_rt_end[0] = self.obj_t[0]
        self.hole_rt_end[1] = self.obj_t[1]
        self.hole_rt_end[2] = self.obj_t[2] + 0.057
        t2[:] = self.hole_rt_end
        t2[2] += 0.01
        R2 = R1.copy()
        planner1 = self.cal_planner(t1, R1, t2, R2, time1)

        time2 = 4.0
        t3 = t2.copy()
        t3[2] = t2[2] - 0.01
        R3 = R2.copy()
        planner2 = self.cal_planner(t2, R2, t3, R3, time2)

        time3 = 0.5
        t4 = t3.copy()
        R4 = R3.copy()
        planner3 = self.cal_planner(t3, R3, t4, R4, time3)

        time_array = np.array([0, time0, time1, time2, time3])
        planner_array = [planner0, planner1, planner2, planner3]
        total_time = np.sum(time_array)

        time_step_num = round(total_time * self._timeStep) + 1
        every_step_num = 20
        every_epoch_num = time_step_num // every_step_num
        times = np.linspace(0, total_time, time_step_num)
        desired_poses = np.zeros((time_step_num, self.numdof))

        states = np.zeros((every_epoch_num, 3))
        actions = np.zeros((every_epoch_num, 3))
        point_clouds = np.zeros((every_epoch_num, self.num_points, 6))

        time_cumsum = np.cumsum(time_array)
        joint_indices = []
        for i in range(self.numjoint):
            joint_info = p.getJointInfo(self.ur5_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # 过滤掉固定关节
                joint_indices.append(i)
        joint_position = [p.getJointState(self.ur5_id, i)[0] for i in joint_indices]

        for i, timei in enumerate(times):
            for j in range(len(time_cumsum)):
                if timei < time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    target_t = planner_interpolate.t
                    target_r = planner_interpolate.R
                    target_r = get_quaternion_from_matrix(target_r)

                    target_r_xyzw = self.shift_array(list(target_r))
                    # ------------------------------------------求解器-------------------------------------------------------
                    self.target_joint_angles = p.calculateInverseKinematics(
                                                        bodyUniqueId=self.ur5_id,
                                                        endEffectorLinkIndex=7,
                                                        targetPosition=target_t,
                                                        targetOrientation=target_r_xyzw,
                                                        restPoses=joint_position,
                                                        jointDamping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                                                        physicsClientId=self.physicsClient_use)

                    joint_position = self.target_joint_angles
                    break

            desired_poses[i, :] = joint_position
            """ test desired_pose """
            # self.control_jointsArray_to_target(self.ur5_id, list(desired_poses[i, :]),
            #                                    joint_indices, physicsClientId=self.physicsClient_use)
            """ test desired_pose """

        data_num = 0
        time_num = 0
        while p.isConnected():
            step_start = time.time()

            if time_num % every_step_num == 0:
                obs = self.get_observation()
                point_cloud = obs['point_cloud']

                """ visualize point cloud """
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                # pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255.0)
                # o3d.visualization.draw_geometries([pcd])
                """ visualize point cloud """

                joints_info = self.setup_control_joint(self.ur5_id, self.robot_control_joint_name)
                robot_state_position = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
                # robot_state_transition = self.Visualize_rotation_center(robot_state_position[4], robot_state_position[5],
                #                                                         relative_offset=[0.055, 0, 0], UI=False)
                joint_state = robot_state_position[0]

                # joint_state, _, _ = self.getJointStates(self.ur5_id, self.control_joint_ids)
                self.control_joints_to_target(self.ur5_id_plan, list(desired_poses[time_num, :]), self.control_joint_ids, physicsClientId=self.physicsClient_plan)
                robot_position_plan = p.getLinkState(self.ur5_id_plan, self.ur5EndEffectorIndex, computeForwardKinematics=1, physicsClientId=self.physicsClient_plan)
                # robot_transition_plan = self.Visualize_rotation_center(robot_position_plan[4], robot_position_plan[5],
                #                                                        relative_offset=[0.055, 0, 0], UI=False)
                peg_position = robot_position_plan[0]

                state = np.array(joint_state) # 3
                action = peg_position

                states[data_num, ...] = state
                actions[data_num, ...] = action
                point_clouds[data_num, ...] = point_cloud
                data_num += 1

            time_num += 1
            if time_num >= time_step_num - every_step_num:
                break

            self.control_joints_to_target(self.ur5_id, desired_poses[time_num, :], self.control_joint_ids,
                                          physicsClientId=self.physicsClient_use)

            time_until_next_step = 1/self._timeStep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        # p.disconnect(physicsClientId=self.physicsClient_plan)
        # p.disconnect(physicsClientId=self.physicsClient_use)

        return {
            'states': states,
            'actions': actions,
            'point_clouds': point_clouds
        }

    def step(self, action):
        dt = 1.0 / self._timeStep
        n_steps = self._timeStep // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # ------------------------------------------求解器-------------------------------------------------------

                self.target_joint = p.calculateInverseKinematics(
                    bodyUniqueId=self.ur5_id,
                    endEffectorLinkIndex=7,
                    targetPosition=action,
                    targetOrientation=list(self.peg_orientation),
                    jointDamping=[0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                    physicsClientId=self.physicsClient_use)

                # self.control_jointsArray_to_target(self.ur5_id, list(self.target_joint),
                #                                    [1,2,3,4,5,6], physicsClientId=self.physicsClient_use)
                self.control_joints_to_target(self.ur5_id, list(self.target_joint),
                                                   [1,2,3,4,5,6], physicsClientId=self.physicsClient_use)


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
            forcemaxforce = 50
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
        self.wait_n_steps(120, physicsClientId)

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
        self.camera_Position = p.getLinkState(self.ur5_id, 10, computeForwardKinematics=1)[0]
        self.camera_Orientation = p.getLinkState(self.ur5_id, 10, computeForwardKinematics=1)[1]
        self.goalPosition1.update(self.camera_Position, self.camera_Orientation)
        self.cube_position = p.getBasePositionAndOrientation(self.tool_id[0])[0]
        self.camera_in_world = [-0.6, 0.2, 0.7]
        self.view_matrix = p.computeViewMatrix(cameraEyePosition = [self.camera_in_world[0],
                                                                self.camera_in_world[1],
                                                                self.camera_in_world[2]],
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


        # intrinsics of the camera
        self.fov = 80
        aspect = float(self.image_width / self.image_height)
        near = 0.001
        far = 200.0
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
        assert (images[0] == self.image_width)
        assert (images[1] == self.image_height)
        rgb_image = np.reshape(images[2], (self.image_height, self.image_width, 4)) * 1. / 255.
        rgb_image_3_channel = np.uint8(rgb_image[:, :, :3] * 255)

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
        assert (rgb_image_3_channel.shape == (self.image_height, self.image_width, 3))

        """ save images """
        cv2.imwrite("./eye_in_hand_rgb.jpg", rgb_image_3_channel)
        cv2.imwrite("./eye_in_hand_depth.jpg", depth_image_uint8)

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

        obs = {
            'agent_pos': self.peg_position,
            'point_cloud': sampled_points
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