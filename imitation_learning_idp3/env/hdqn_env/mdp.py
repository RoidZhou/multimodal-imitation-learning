from gym import spaces
import random
class StochasticMDPEnv:

    def __init__(self):
        self.visited_six = False
        self.current_state = 2
        # number of actions (left: 0, right: 1)
        self.nA = 2
        # number of states
        self.nS = 6

    def reset(self):
        self.visited_six = False
        self.current_state = 2
        return self.current_state

    def step(self, action):
        if self.current_state != 1:
            # If "right" selected
            if action == 1:
                if random.random() < 0.5 and self.current_state < 6:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
            if self.current_state == 6:
                self.visited_six = True
        if self.current_state == 1:
            if self.visited_six:
                return self.current_state, 1.00, True, {}
            else:
                return self.current_state, 1.00/100.00, True, {}
        else:
            return self.current_state, 0.0, False, {}

import skfuzzy as fuzz
from tensorboardX import SummaryWriter
import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data
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
from numpy.ma.core import argmin
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


class UR5Env:
    metadata = {'render.modes': ['human']}
    def __init__(self, cfg, render=True):
        super().__init__()
        self.log =[]
        self.action_dim = cfg.action.shape[0]

        self.randm_num = 1
        self.mointor_force_torque = np.zeros((2, 60))
        self.neibu = False
        self.sucessful_number = 0 # 迭代成功的次数
        self.all_number = 0  # 迭代次数
        self.epsiode_timesteps = 0
        self.max_steps_one_episode = 50#26
        self.goal_cont = 0
        self.step_counter = 0
        self.quat_rot_err = np.zeros(4)
        self.current_twist_lin = np.zeros((3, 1))
        self.current_twist_ang = np.zeros((3, 1))
        self.arm_desired_twist_ = np.mat(np.zeros((6, 1)))
        self.arm_desired_position_ = np.mat(np.zeros((3, 1)))
        self.arm_max_acc_ = 1
        self.arm_max_acc_r = 10
        self.duration = 0.2
        self.obj_t = np.zeros(3)
        self.num_points = 4096 * 2
        self.control_hz = 100
        self.prev_heght = 0
        self.prev_angle_err = 30
        self.goal_0_reach = 0
        self.goal_1_reach = 0
        self.goal = 0
        self.arrive_orien_num = 0
        self.first_pose_to_depth_flag = 1
        self.impedance_depth = 0.0
        self.impedance_arrive = 0

        self.Visualize_rotation_center_UI = DebugAxes()  # 可视化旋转中心
        self.goalPosition1 = DebugAxes()  # 可视化 eelink 坐标
        self.goalPosition_eelink = DebugAxes()  # 可视化 eelink 坐标
        self.goalPosition_hole = DebugAxes()  # 可视化 eelink 坐标
        self.writer = SummaryWriter('./HDQN_peg/logs')

        self.image_width = 320
        self.image_height = 240

        # 机械臂实际执行频率，ur5真实通讯频率是120hz
        self._timeStep = 240
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
        init_end_orien = np.random.uniform(-0.5, 0.5)
        self.init_joint_val =[0.21195291679571792, -1.2151152721500111, -2.102115573329094, # 0.10195291679571792
                              -1.4221581329978485, 1.5657963241241731, 0.17195291700664328+init_end_orien]#标准垂直姿态,不接触桌面
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
            self.physicsClient_use = p.connect(p.DIRECT)
            self.physicsClient_plan = p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)
        # 设定界面显示视角
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        p.setTimeStep(1/240)

        # -----------------------------------------------------------------------添加模型-----------------------------------------------------------------------------------------------
        # 添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.tool_id = p.loadSDF("./assert/ur_description/urdf/platform/urdf/platform.sdf")
        """ 用于测试恒力跟踪"""
        p.changeDynamics(self.tool_id[0], -1,
                         lateralFriction=0.1, spinningFriction=0.1, rollingFriction=0, frictionAnchor=True)

        #  直的
        p.resetBasePositionAndOrientation(self.tool_id[0], [-0.4 + 0.05, 0.1 - 0.05, 0.32],
                                          p.getQuaternionFromEuler([0, 0, 0]))#100宽度*100高*孔21
        self.init_height = 0.32+0.08
        self.goalPosition =[-0.4+0.05, 0.1-0.05, 0.32+0.08]

        # 添加桌子模型
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[-1.01, 0, -0.315])

        # 添加机器人模型
        self.ur5_id = p.loadURDF(
            "./assert/ur_description/urdf/ur5_robot_sensor_eelink_triangle.urdf",
            basePosition=[0, 0, 0.1], flags=9)
        self.ur5_id_plan = p.loadURDF(
            "./assert/ur_description/urdf/ur5_robot_sensor_eelink_triangle.urdf",
            basePosition=[0, 0, 0.1], flags=9, physicsClientId = self.physicsClient_plan)
        self.ur5EndEffectorIndex = 7
        contact_stiffness = 500  # 较低刚度，柔软接触
        contact_damping = 100  # 适中阻尼

        self.set_link_stiffness(self.ur5_id, self.ur5EndEffectorIndex, contact_stiffness, contact_damping,
                                self.physicsClient_use)

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
        self.obj_r = hole_orientation
        self.goalPosition_hole.update(hole_position, hole_orientation)


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
        self.translational_stiffness = 10
        self.rotational_stiffness = 10
        self.translational_damping = 400
        self.translational_damping = 400

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
        self.goal_0_reach = 0
        self.goal_1_reach = 0
        self.goal_cont=0
        self.randm_num += 1
        self.solve_steps = 0
        self.int_falg = True
        self.arrive_orien_num = 0
        self.first_pose_to_depth_flag = 1
        self.impedance_depth = 0.0
        self.impedance_arrive = 0
        self.mointor_force_torque = np.zeros((2, 60))
        p.enableJointForceTorqueSensor(self.ur5_id, 7)
        p.stepSimulation()

        # --------------------------------------- 重置关节至初始状态------------------------------------这里有坑，p.resetJointState与p.setTimeStep()会导致初始姿态偏移
        init_end_orien = np.random.uniform(-1, 1)
        init_joint0_orien = np.random.uniform(-0.1, 0.1)
        self.init_joint_val[5] += init_end_orien
        # self.init_joint_val[0] += init_joint0_orien
        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.init_joint_val[i])
            p.resetJointState(bodyUniqueId=self.ur5_id_plan, jointIndex=i + 1, targetValue=self.init_joint_val[i], physicsClientId=self.physicsClient_plan)

        self.hole_up_end = np.zeros(3)
        self.hole_up_end[0] = self.obj_t[0] - 0.0016
        self.hole_up_end[1] = self.obj_t[1] - 0.027
        self.hole_up_end[2] = self.obj_t[2] + 0.115
        self.target_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_id,
            endEffectorLinkIndex=7,
            targetPosition=self.hole_up_end,
            jointDamping=[0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            physicsClientId=self.physicsClient_use)
        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.target_joint_angles[i])
            p.resetJointState(bodyUniqueId=self.ur5_id_plan, jointIndex=i + 1, targetValue=self.target_joint_angles[i], physicsClientId=self.physicsClient_plan)

        # set hole terminal position and orientation
        self.hole_true_orientation = Rotation.from_euler('xyz', [90, 90, -30], degrees=True).as_quat()  # 默认固定孔的姿态
        self.hole_terminal_orientation = np.array(self.hole_true_orientation)
        self.hole_terminal_position = np.array(self.hole_up_end)
        self.hole_terminal_position[2] -= 0.06


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
        robot_current_state = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
        self.robot_current_orientation = np.array(robot_current_state[1])
        self.robot_current_position = np.array(robot_current_state[0])
        self.robot_reset_position = self.robot_current_position
        self.robot_reset_orientation = self.robot_current_orientation

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

    def step(self, action):
        self.step_counter += 1
        dt = 1.0 / self._timeStep
        n_steps = self._timeStep // self.control_hz
        if action is not None:
            # action_position = self.hole_up_end
            robot_prev_state = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
            self.robot_prev_orientation = np.array(robot_prev_state[1])
            self.robot_prev_position = np.array(robot_prev_state[0])
            position_prev_err = np.sqrt(np.sum((self.hole_terminal_position - self.robot_prev_position) ** 2))
            orientation_prev_err = np.sqrt(np.sum((self.hole_terminal_orientation - self.robot_prev_orientation) ** 2))
            self.prev_height = self.robot_prev_position[2]

            action_position = np.array(action[0:3])
            action_orientation_6d = np.array(action[3:self.action_dim])
            # action_orientation = _6d_to_quaternion(action_orientation_6d)
            for i in range(n_steps):
                # ------------------------------------------求解器-------------------------------------------------------
                if self.goal == 1:
                    self.apply_hybrid_controller(
                        np.concatenate((action_position, action_orientation_6d)),
                        physicsClientId=self.physicsClient_use)
                else:
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
            # print("---- force ---", current_force)

            if np.linalg.norm(current_force) < -1:
                robot_state_position[2] -= 0.009
                if self.goal == 1:
                    self.apply_hybrid_controller(
                        np.concatenate((robot_state_position, robot_srate_orientation)),
                        physicsClientId=self.physicsClient_use, if_test = True)
                else:
                    self.target_joint = p.calculateInverseKinematics(
                        bodyUniqueId=self.ur5_id,
                        endEffectorLinkIndex=7,
                        targetPosition=list(robot_state_position),
                        targetOrientation=list(robot_srate_orientation),
                        jointDamping=[0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                        physicsClientId=self.physicsClient_use)
                    self.control_joints_to_target(self.ur5_id, list(self.target_joint),
                                                  [1, 2, 3, 4, 5, 6], physicsClientId=self.physicsClient_use)

            robot_current_state = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
            self.robot_current_orientation = np.array(robot_current_state[1])
            self.robot_current_position = np.array(robot_current_state[0])

        pose_err = action_position - self.robot_current_position
        observation = self.get_observation()
        # if pose_err[2] > 0:
            # print("peg move down", pose_err[2])
        done = False
        info = None
        reward = 0

        pose_err = self.robot_current_position - self.robot_prev_position
        insert_depth = self.robot_reset_position[2] - self.robot_current_position[2]
        orientation_err = np.dot(self.robot_prev_orientation, self.robot_current_orientation)
        orien_angle_err = 2*np.arccos(orientation_err)
        angle_err = orien_angle_err * 180 / np.pi

        angles = []
        angles_select = [-150, -30, 90, 210]
        for i in range(4):
            hole_orientation = Rotation.from_euler('xyz', [90, 90, -150+120*i], degrees=True).as_quat()  # 默认固定孔的姿态

            robot_state_position = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
            state_orientation = robot_state_position[1]
            orientation_err = np.dot(hole_orientation, state_orientation)
            orien_angle_err = 2*np.arccos(orientation_err)
            angle_err = orien_angle_err*180/np.pi
            angles.append(angle_err)

        # if insert_depth > 0.007:
        #     done = True
        #     reward = 10
        #     return observation, reward, done, info

        if self.goal == 0:
            if min(angles) < 4 and self.goal_0_reach == 0:
                self.goal_0_reach = 1
                reward = 1
            if insert_depth > 0.004 and self.goal_0_reach == 0:
                self.goal_0_reach = 1
                reward = 1
            if insert_depth > 0.004 and self.goal_0_reach == 1:
                done = True
                reward = 3

        if self.goal == 1:
            # if self.goal_0_reach == 0:
            #     done = True
            #     reward = 0
            if self.orien_failed == 1:
                reward = 0
                done = True
            if insert_depth > 0.004 and self.goal_1_reach == 0:
                # done = True
                reward = 2
                self.goal_1_reach = 1
            if insert_depth > 0.007:
                done = True
                reward = 3
            if (abs(current_force[0]) > 20 or abs(current_force[1]) > 20 or abs(current_force[2]) > 20) and self.goal_0_reach == 1:
                reward = -0
                done = True
            elif abs(current_force[2]) > 100 and self.goal_0_reach == 1 and self.goal_1_reach == 0:
                reward = -0

        if self.goal == 2:
            if self.goal_0_reach == 0 or self.goal_1_reach == 0:
                done = True
                reward = 0
            if insert_depth > 0.007:
                done = True
                reward = 3
        if self.step_counter > 500:
            done = True
            print("step counter arrive")

        # ----  以下全部注释  -----
        # orientation_err = np.dot(self.hole_terminal_orientation, self.robot_current_orientation)
        # orien_angle_err = 2*np.arccos(orientation_err)
        # angle_err = orien_angle_err*180/np.pi
        # target_angle = self.find_nearest_target(angle_err)
        # if self.is_angle_close(angle_err,tolerance=1):
        #     done = True
        #     self.goal_0_reach = 1
        #     reward = 10
        #
        #     if current_force[2] < 100:
        #         robot_state_position[2] -= 0.009
        #
        #         self.target_joint = p.calculateInverseKinematics(
        #             bodyUniqueId=self.ur5_id,
        #             endEffectorLinkIndex=7,
        #             targetPosition=list(robot_state_position),
        #             targetOrientation=list(robot_srate_orientation),
        #             jointDamping=[0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
        #             physicsClientId=self.physicsClient_use)
        #         self.control_joints_to_target(self.ur5_id, list(self.target_joint),
        #                                       [1, 2, 3, 4, 5, 6], physicsClientId=self.physicsClient_use)
        #
        #     robot_current_state = p.getLinkState(self.ur5_id, self.ur5EndEffectorIndex, computeForwardKinematics=1)
        #     self.robot_current_orientation = np.array(robot_current_state[1])
        #     self.robot_current_position = np.array(robot_current_state[0])
        #
        #     pose_err = action_position - self.robot_current_position
        #     observation = self.get_observation()
        #
        #     return observation, reward, done, info
        # else:
        #     curr_angle_err = abs(target_angle - angle_err)
        #     if self.prev_angle_err is None:
        #         self.prev_angle_err = curr_angle_err
        #     reward = self.prev_angle_err - curr_angle_err
        #     # reward = 0
        #
        #
        # position_curr_err = np.sqrt(np.sum((self.hole_terminal_position - self.robot_current_position)**2))
        # total_pose_err = -(position_prev_err - position_curr_err)
        # total_orien_err = (orientation_prev_err - orientation_err)
        # self.curr_height = self.robot_current_position[2]
        # self.height_err = self.prev_height - self.curr_height
        # # if total_pose_err>0 or total_orien_err>0:
        # #     done = True
        # FT = self.getForceTorque()
        # current_force = np.array([FT[0], FT[1], FT[2]])
        # if np.max(abs(current_force)) > 10000:
        #     done = True
        #     print("max force: ", current_force)
        # if self.step_counter > 100:
        #     done = True
        #     print("step counter arrive")
        # if (self.height_err > 0 and current_force[2]< 50) :
        #     reward = 1
        #     if position_curr_err < 0.055:
        #         done = True
        #         reward = 10
        #
        # if current_force[2] > 50:
        #     done = True
        #
        # self.prev_angle_err = curr_angle_err

        return observation, reward, done, info

    def set_link_stiffness(self, robot_id, link_index, contact_stiffness, contact_damping, physics_client_id):
        """
        设置机器人 link 的接触刚度和阻尼。

        参数:
            robot_id: 机器人 ID
            link_index: link 索引（例如末端执行器的索引）
            contact_stiffness: 接触刚度 (N/m)
            contact_damping: 接触阻尼 (N·s/m)
            physics_client_id: 物理客户端 ID
        """
        p.changeDynamics(
            bodyUniqueId=robot_id,
            linkIndex=link_index,
            contactStiffness=contact_stiffness,
            contactDamping=contact_damping,
            physicsClientId=physics_client_id
        )

    def shift_array(self, arr):
        if len(arr) <= 1:
            return arr
        first_ele = arr.pop(0)
        arr.append(first_ele)

        return arr

    def is_almost_integer(self, k, epsilon=0.05):
        """检查 k 是否在某个整数的附近（允许误差 epsilon）"""
        n = round(k)  # 最近的整数
        return abs(k - n) <= epsilon

    def is_angle_close(self, angle_err, tolerance=1):
        """
        检查 angle_err 是否以 ±tolerance 的误差靠近目标角度 [0, 120, 240, 360]。
        - angle_err: 当前角度（0~360）
        - tolerance: 允许的误差（默认 ±1°）
        - 返回: True（靠近）或 False（不靠近）
        """
        target_angles = np.array([0, 120, 240, 360])  # 目标角度列表

        # 计算 angle_err 与所有 target_angles 的绝对差值（考虑 360° 环绕）
        diffs = np.abs((angle_err - target_angles + 180) % 360 - 180)

        # 取最小差值
        min_diff = np.min(diffs)

        # 判断是否在容差范围内
        return min_diff <= tolerance

    def find_nearest_target(self, theta):
        """找到离 theta 最近的目标角度（30°, 90°, 150°, ...）"""
        targets = np.array([0, 120, 240, 360])
        nearest_target = targets[np.argmin(np.abs(targets - theta))]
        return nearest_target


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
                                    positionGain=0.1,
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

        # peg_orientation_6d = quaternion_to_6d(self.peg_orientation, order='xyzw')
        action_position = np.array(self.peg_position)
        action_orientation = np.array(self.peg_orientation)
        actions[0:3] = action_position
        actions[3:self.action_dim] = action_orientation
        # actions[0:self.action_dim] = self.peg_orientation
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

    def cart_to_base_force_torque(self, xyz_force, xyz_torque, position, orientation):
        """
        将末端坐标系下的六维力转换到基坐标系。

        参数：
            xyz_force: 末端坐标系下的力 (3x1 矩阵)。
            xyz_torque: 末端坐标系下的力矩 (3x1 矩阵)。
            orientation: 末端坐标系的姿态（四元数或欧拉角）。
            position: 末端坐标系的位置 (3x1 向量)。

        返回：
            force_torque_base: 基坐标系下的六维力 (6x1 矩阵)。
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        # 将姿态转换为旋转矩阵
        if len(orientation) == 4:  # 四元数
            rotation = R.from_quat(orientation)
        else:  # 欧拉角
            rotation = R.from_euler('xyz', orientation, degrees=False)
        rotation_matrix = rotation.as_matrix()

        # 转换力
        force_base = rotation_matrix @ xyz_force

        # 转换力矩
        r = np.array(position)
        torque_base = rotation_matrix @ xyz_torque

        # 组合结果
        force_torque_base = np.vstack((force_base, torque_base))
        return force_torque_base

    def impedance_controller(self, action, physicsClientId, if_test):
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
        desired_force_z = -1  # N
        desired_force_xy = 0.0  # N
        desired_force_rz = 0.0
        Kp_force = 0.1  # 比例增益
        Ki_force = 0.02  # 积分增益
        Kd_force = 0.01  # 微分增益
        integral_force_error_z = 0.0
        previous_force_error_z = 0.0
        delta_z_max = 20
        # 获取当前的关节力和力矩
        FT = self.getForceTorque()
        Fext = [FT[0], FT[1], FT[2]]
        Text = [FT[3], FT[4], FT[5]]
        # Text = [0, 0, 0]
        # 获取当前的关节状态
        self.current_Position = p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[4]
        self.current_Orientation = np.array(p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[5])
        force_torque_base = self.cart_to_base_force_torque(np.array(Fext), np.array(Text), self.current_Position, self.current_Orientation)
        self.force_x = force_torque_base[0][0]
        self.force_y = force_torque_base[0][1]
        self.force_z = force_torque_base[0][2]

        self.Torque_x = force_torque_base[1][0]
        self.Torque_y = force_torque_base[1][1]
        self.Torque_z = force_torque_base[1][2]

        action_to_target_pose[0] = action[0] + self.zero_Position[0]
        action_to_target_pose[1] = action[1] + self.zero_Position[1]
        action_to_target_pose[2] = action[2] + self.zero_Position[2]

        action_to_target_orie[0] = action[3]
        action_to_target_orie[1] = action[4]
        action_to_target_orie[2] = action[5]
        action_to_target_orie[3] = action[6]

        wrench_z = force_torque_base[0][2]
        # print("force_torque_base : ", force_torque_base)
        if abs(self.force_z) > 1:
            print("force_z")
        self.force_error_z = desired_force_z - wrench_z
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
        pos_z_adjustment = pos_z_adjustment / 10  # 手动补偿
        if wrench_z > 0.1:
            # print("wrench_z : ", wrench_z)
            pos_z_adjustment = pos_z_adjustment
        for i in range(1):
            # if Fext[1] > 0.1 or Text[2] > 0.001 :
            #     desired_force_rz = -0.1
            #     desired_force_xy = -0.1
            force_torque_base[0][:] = [x if abs(x)>=0.1 else 0.0 for x in force_torque_base[0][:]]
            force_torque_base[1][:] = [x if abs(x)>=0.1 else 0.0 for x in force_torque_base[1][:]]
            force_external = np.mat([[force_torque_base[0][0] - desired_force_xy], [force_torque_base[0][1] - desired_force_xy], [0.0], [force_torque_base[1][0] - desired_force_rz], [force_torque_base[1][1] - desired_force_rz], [force_torque_base[1][2] - desired_force_rz]])
            # force_external = np.mat([[0], [Fext[1]], [0], [0], [0], [0]])

            # 获取当前的关节状态
            # self.current_Position = p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[4]
            # self.current_Orientation = np.array(p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[5])
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
            if if_test == True:
                quat_rot_err_tmp = quat_rot_err_ / 1 # for test, is related to the size of target pose that is input
            else:
                quat_rot_err_tmp = quat_rot_err_ / 1
            self.rz_tmp = quat_rot_err_[2]
            # quat_rot_err_tmp = quat_rot_err_tmp * 180 / np.pi
            # print("quat_rot_err_tmp: ", quat_rot_err_tmp)

            if self.pre_dz == 0.0:
                self.deta_dz = self.current_Position[2]
                self.pre_dz = self.current_Position[2]
            else:
                self.deta_dz = self.pre_dz - self.current_Position[2]
                self.pre_dz = self.current_Position[2]
            self.insert_depth += self.deta_dz
            # Position error
            if abs(action_to_current[0]) < 1e-4:
                action_to_current[0] = 0.0
            if abs(action_to_current[1]) < 1e-4:
                action_to_current[1] = 0.0
            if abs(action_to_current[2]) < 1e-4:
                action_to_current[2] = 0.0
            if abs(quat_rot_err_[0]) < 1e-2:
                quat_rot_err_tmp[0] = 0.0
            if abs(quat_rot_err_[1]) < 1e-2:
                quat_rot_err_tmp[1] = 0.0
            if abs(quat_rot_err_[2]) < 1e-2:
                quat_rot_err_tmp[2] = 0.0
            self.dx = action_to_current[0] * 100.0
            self.dy = action_to_current[1] * 100.0
            self.dz = action_to_current[2] * 100.0
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

    def reset_controller(self):
        self.arm_desired_twist_ = np.mat(np.zeros((6, 1)))
        self.arm_desired_position_ = np.mat(np.zeros((3, 1)))
        self.desire_control_error_integral_ = np.mat(np.zeros((6, 1)))
        self.period = 0.5
        self.arm_max_vel = 0.001
        self.arm_max_acc = 0.001
        # selection matrix
        self.force_matrix_ = np.diag(np.array([0, 0, 1, 0, 0, 0]))
        self.position_matrix_ = np.diag(np.array([1, 1, 0, 0, 0, 0]))
        self.force_matrix = np.mat(self.force_matrix_)
        self.position_matrix = np.mat(self.position_matrix_)
        self.pre_dz = 0.0
        self.insert_depth = 0.0
        self.solve_steps = 0
        self.orien_failed = 0
        self.solve_steps = 0

    def apply_hybrid_controller(self, action, physicsClientId, if_test=False):
        """ Make a step in simulation """
        position_arrive = False
        """
        k=40,d=28.28,m=10
        """
        self.translational_stiffness = 40
        self.rotational_stiffness = 40
        self.translational_damping = 28.28
        self.translational_damping = 28.28
        self.duration = 0.002
        self.zero_Orientation[0] = action[3]
        self.zero_Orientation[1] = action[4]
        self.zero_Orientation[2] = action[5]
        self.zero_Orientation[3] = action[6]
        self.reset_controller()
        while 1:
            desired_twist, deta_desired_twist = self.impedance_controller(action, physicsClientId, if_test=False)
            desired_twist = np.array(desired_twist)
            # print("desired_twist: ", desired_twist)
            self.send_commands_to_robot(desired_twist[0], desired_twist[1], desired_twist[2], desired_twist[3], desired_twist[4], desired_twist[5], physicsClientId)
            current_Position = p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[4]
            current_Orientation = np.array(p.getLinkState(self.ur5_id, 7, physicsClientId=physicsClientId)[5])

            self.solve_steps = self.solve_steps + 1
            # print("self.force_error_z : ", self.force_error_z)
            joint_positions = p.getJointState(self.ur5_id, 6, physicsClientId=physicsClientId)[0]
            # print("self.position_error_y : ", self.position_error_y)
            self.writer.add_scalars("force_x",
                                   {"force_x": self.force_x}, self.solve_steps)
            self.writer.add_scalars("force_y",
                                   {"force_y": self.force_y}, self.solve_steps)
            self.writer.add_scalars("force_z",
                                   {"force_z": self.force_z}, self.solve_steps)
            self.writer.add_scalars("Torque_x",
                                   {"Torque_x": self.Torque_x}, self.solve_steps)
            self.writer.add_scalars("Torque_y",
                                   {"Torque_y": self.Torque_y}, self.solve_steps)
            self.writer.add_scalars("Torque_z",
                                   {"Torque_z": self.Torque_z}, self.solve_steps)
            # if (self.rz < 5e-6 and self.dz < 5e-6) or self.is_in_range(1, self.orientation_err, 5e-6) or self.angle_err < 0.1 or self.angle_err==None:
            if if_test == True:
                if (abs(self.rz) < 5e-2 and abs(self.deta_dz) < 5e-8) or self.insert_depth > 0.435 or self.solve_steps > 3000:
                # if (abs(self.rz) < 5e-6):
                    print("force err success")
                    break
            else:
                if self.orien_failed == 1:
                    break
                if (abs(self.rz_tmp) < 5e-3) or self.insert_depth > 0.435 or self.solve_steps > 1000:
                    # if (abs(self.rz) < 5e-6):
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
            if joint_index == 6:
                p.setJointMotorControl2(self.ur5_id, joint_index, p.VELOCITY_CONTROL, velocityGain=0.5, targetVelocity=joint_velocities[i], force=500, physicsClientId=physicsClientId)
            else:
                p.setJointMotorControl2(self.ur5_id, joint_index, p.VELOCITY_CONTROL, targetVelocity=joint_velocities[i], force=100, physicsClientId=physicsClientId)
                # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(10):
            p.stepSimulation()