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
import pyrealsense2 as rs
from math import *
from socket import *
import struct
import transforms3d as tfs
import tf
import minimalmodbus as mm
import serial
import serial.tools.list_ports
from scipy.spatial.transform import Rotation
import rospy
# from cartesian_state_msgs.msg import PoseTwist
from socket import *
from spatialmath import SE3
import spatialmath as sm
sys.path.append('../../../imitation_learning_idp3')
from imitation_learning_idp3.arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner


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

        rot3x3 = Rotation.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


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

    end_orn = Rotation.from_quat(end_orn).as_matrix()
    wcT = np.eye(4)
    # wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos #注意：务必注意自己的变换要求！
    fg = Rotation.from_euler('xyz', relative_euler).as_matrix()
    wcT[:3, :3] = np.matmul(end_orn[:3, :3], fg)
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos

    return wcT


def fix_center_rotation(end_pos, end_orn, relative_offset, relative_euler, dy_M=0.09):
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


class Realsense:
    def __init__(self, width=640, height=480, fps=15):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.intrinsics = None
        self.scale = None
        self.pipeline = None

        self.connect()
        print("camera init")

    def connect(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        cfg = self.pipeline.start(config)
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print(f"device: {cfg.get_device()}")
        print(f"depth_sensor: {cfg.get_device().first_depth_sensor()}")
        print(f"depth_scale: {self.scale}")
        print(f"streams: {cfg.get_streams()}")

        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(rgb_profile)
        print("---------D435 CONNECT!----------")

    def get_data(self):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def plot_image(self):
        # 检查分辨率是否需要调整
        color_image, depth_image = self.get_data()
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        resize_needed = depth_colormap_dim != color_colormap_dim

        # 创建窗口
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        # 实时显示循环
        try:
            while True:
                # 获取新帧
                color_image, depth_image = self.get_data()
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # 调整分辨率（如果需要）
                if resize_needed:
                    images = np.hstack((cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                   interpolation=cv2.INTER_AREA), depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                # 显示图像
                cv2.imshow('RealSense', images)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # 停止管道并关闭窗口
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def get_intrinsics(self, rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera intrinsics:", raw_intrinsics)
        intrinsics = np.array(
            [raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)
        return intrinsics


class socket_TCp_UR30003():
    def __init__(self):
        self.host_name = "192.168.1.102"
        self.port_num = 30003
        self.ClientSocket = socket(AF_INET, SOCK_STREAM)
        self.ClientSocket.connect((self.host_name, self.port_num))

    def UR_30003Script(self, send_data):
        # print(send_data)
        self.ClientSocket.send(send_data.encode('utf8'))

    def UR_30003rt(self, Meaning):

        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d',
               'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
               'softwareOnly2': 'd', 'V main': 'd',
               'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
               'Elbow position': '3d', 'Elbow velocity': '3d'}
        data = self.ClientSocket.recv(1220)
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])
            info, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, info)
        f = 1

        return dic[Meaning]

    def get_current_pose(self):
        tcp_socket = socket(AF_INET, SOCK_STREAM)
        tcp_socket.connect((self.host_name, self.port_num))
        data = tcp_socket.recv(1108)
        # print(len(data))
        position = struct.unpack('!6d', data[444:492])
        orientation = struct.unpack('!6d', data[492:540])
        tcp_socket.close()
        return np.asarray(position), np.asarray(orientation)

    def get_state_cart(self):
        self.tcp_socket = socket(AF_INET, SOCK_STREAM)
        self.tcp_socket.connect((self.host_name, self.port_num))
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')

        self.tcp_socket.close()
        return actual_joint_positions

    def get_state_joint(self):
        self.tcp_socket = socket(AF_INET, SOCK_STREAM)
        self.tcp_socket.connect((self.host_name, self.port_num))
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')

        self.tcp_socket.close()
        return actual_joint_positions

    def parse_tcp_state_data(self, data, subpasckage):
        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d',
               'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
               'softwareOnly2': 'd',
               'V main': 'd',
               'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
               'Elbow position': 'd', 'Elbow velocity': '3d'}
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])  # 计算key对应value的size
            data1, data = data[0:fmtsize], data[fmtsize:]  # 根据size分割数据
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, data1)

        if subpasckage == 'joint_data':  # get joint data
            q_actual_tuple = dic["q actual"]
            joint_data = np.array(q_actual_tuple[1])
            return joint_data
        elif subpasckage == 'cartesian_info':
            Tool_vector_actual = dic["Tool vector actual"]  # get x y z rx ry rz
            cartesian_info = np.array(Tool_vector_actual[1])
            return cartesian_info

    def speed_l(self, xd, a, t, aRot='a'):
        """
            Tool speed eg: speedl([0.5,0.4,0,1.57,0,0], 0.5, 0.5)
            xd: tool speed
            a: tool position acceleration
            t: time
            aRot: tool acceleration 没定义a则使用这个
        """
        self.tcp_socket = socket(AF_INET, SOCK_STREAM)
        self.tcp_socket.connect((self.host_name, self.port_num))
        tcp_command = 'speedl([%f' % xd[0]
        for i in range(1, 6):
            tcp_command = tcp_command + (',%f' % xd[i])
        tcp_command = tcp_command + '],a=%f,t=%f)' % (a, t)
        print(tcp_command)
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

    def movej_offset(self, offset):
        '''TCP_pos:是当前tool的
        '''
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        global pose=get_actual_tcp_pose()
        global P= pose_trans(pose,p[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}])
        # global P2=get_inverse_kin(P)
        movej(P, a=0.05, v=0.25, t=0, r=0)


    end
        '''
        # print(send_data)
        self.UR_30003Script(send_data)  # 30003发送

    def movej(self, offset):
        '''TCP_pos:是当前tool的
        '''
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        # global pose=[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}]
        # popup(pose)
        movej([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], a=0.05, v=0.25, t=0.5, r=0)


    end
            '''
        self.UR_30003Script(send_data)  # 30003发送 t=2.1

    # 直接控制六个轴的速度
    def speedj(self, offset, time=1):
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        speedj([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], 1, {time})
    end
            '''
        self.UR_30003Script(send_data)  # 30003发送

    def speed_j(self, qd, a=0.5, t=5):
        """
            Joint speed eg: speedj([0.2,0.3,0.1,0.05,0,0], 0.5, 0.5)
            qd: joint speed
            a: acceleration
            t: time
        """
        self.tcp_socket = socket(AF_INET, SOCK_STREAM)
        self.tcp_socket.connect((self.host_name, self.port_num))
        tcp_command = "speedj([%f" % qd[0]
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % qd[joint_idx])
        tcp_command = tcp_command + "],a=%f,t=%f)\n" % (a, t)
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

    # 直接控制六个轴的速度
    def speedl(self, offset):
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        speedl([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], 0.5)
    end
            '''
        self.UR_30003Script(send_data)  # 30003发送

    # 控制末端速度类似speedl，但是是欧拉角下
    def speedj_offset(self, offset):

        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        global pose=get_actual_tcp_pose()
        global P= pose_trans(pose,p[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}])
        global P2=get_inverse_kin(pose)
        global P3=get_inverse_kin(P)
        global P4=[P3[0]-P2[0],P3[1]-P2[1],P3[2]-P2[2],P3[3]-P2[3],P3[4]-P2
        [4],P3[5]-P2[5]]
        speedj(P4, 0.2,0.5)
    end
            '''
        # print(send_data)
        self.UR_30003Script(send_data)  # 30003发送

        # [-0.08972922650022963, -1.6219628747300388, -2.0232094758251753, -1.0464530101386091, 1.5717372302313297,
        #  -0.08981790491708695]


class Forceusb():
    '''
    modbus通讯，注册并获取力控数值
    有做零位减法，每次初始化就是置零的过程
    '''

    def __init__(self):
        self.BAUDRATE = 19200
        self.BYTESIZE = 8
        self.PARITY = "N"
        self.STOPBITS = 1
        self.TIMEOUT = 0.2
        self.PORTNAME = self.serial_ports()
        self.SLAVEADDRESS = 9
        self.ser = serial.Serial(port=self.PORTNAME, baudrate=self.BAUDRATE, bytesize=self.BYTESIZE, parity=self.PARITY,
                                 stopbits=self.STOPBITS, timeout=self.TIMEOUT)
        self.packet = bytearray()
        self.sendCount = 0
        while self.sendCount < 50:
            self.packet.append(0xff)
            self.sendCount = self.sendCount + 1
        self.ser.write(self.packet)
        self.ser.close()
        # Communication setup
        mm.BAUDRATE = self.BAUDRATE
        mm.BYTESIZE = self.BYTESIZE
        mm.PARITY = self.PARITY
        mm.STOPBITS = self.STOPBITS
        mm.TIMEOUT = self.TIMEOUT
        self.ft300 = mm.Instrument(self.PORTNAME, slaveaddress=self.SLAVEADDRESS)
        self.registers = self.ft300.read_registers(180, 6)
        # Save measured values at rest. Those values are use to make the zero of the sensor.
        self.fxZero = self.forceConverter(self.registers[0])
        self.fyZero = self.forceConverter(self.registers[1])
        self.fzZero = self.forceConverter(self.registers[2])
        self.txZero = self.torqueConverter(self.registers[3])
        self.tyZero = self.torqueConverter(self.registers[4])
        self.tzZero = self.torqueConverter(self.registers[5])

        self.ft_data = []
        self.num = 0

    def close(self):
        self.ft300.serial.close()

    def serial_ports(self):  # 自动寻找端口
        ports = list(serial.tools.list_ports.comports())
        for port_no, description, address in ports:
            if 'USB' in description:
                return port_no

    def forceConverter(self, forceRegisterValue):
        """Return the force corresponding to force register value.

        input:
            forceRegisterValue: Value of the force register

        output:
            force: force corresponding to force register value in N
        """
        force = 0
        forceRegisterBin = bin(forceRegisterValue)[2:]
        forceRegisterBin = "0" * (16 - len(forceRegisterBin)) + forceRegisterBin
        if forceRegisterBin[0] == "1":
            # negative force
            force = -1 * (int("1111111111111111", 2) - int(forceRegisterBin, 2) + 1) / 100
        else:
            # positive force
            force = int(forceRegisterBin, 2) / 100
        return force

    def torqueConverter(self, torqueRegisterValue):
        """Return the torque corresponding to torque register value.

        input:
            torqueRegisterValue: Value of the torque register

        output:
            torque: torque corresponding to force register value in N.m
        """
        torque = 0

        torqueRegisterBin = bin(torqueRegisterValue)[2:]
        torqueRegisterBin = "0" * (16 - len(torqueRegisterBin)) + torqueRegisterBin
        if torqueRegisterBin[0] == "1":
            # negative force
            # torque=-1*(int("1111111111111111",2)-int(torqueRegisterBin[1:],2)+1)/1000
            torque = -1 * (int("1111111111111111", 2) - int(torqueRegisterBin, 2) + 1) / 1000
        else:
            # positive force
            torque = int(torqueRegisterBin, 2) / 1000
        return torque

    def SN(self, snValue):
        pass

    def get_current_ft(self):
        """
        获得力传感器数值
        """
        registers = self.ft300.read_registers(180, 6)
        fx = round(self.forceConverter(registers[0]) - self.fxZero, 2)
        fy = round(self.forceConverter(registers[1]) - self.fyZero, 2)
        fz = round(self.forceConverter(registers[2]) - self.fzZero, 2)
        tx = round(self.torqueConverter(registers[3]) - self.txZero, 2)
        ty = round(self.torqueConverter(registers[4]) - self.tyZero, 2)
        tz = round(self.torqueConverter(registers[5]) - self.tzZero, 2)
        ft = [fx, fy, fz, tx, ty, tz]
        self.ft_data.append(ft)
        # print("ft", ft[3:6])
        return ft


class Real_UR5_robot():
    metadata = {'render.modes': ['human']}

    def __init__(self, render=True):
        super().__init__()

        self.Fusb = Forceusb()
        self.real_robot_connect = socket_TCp_UR30003()  # 机械臂建立链接
        self.rscamera = Realsense()
        self.ft_compenstate = np.array([0, 0, 0, 0, 0, 0])
        self.Visualize_rotation_center_UI = DebugAxes()  # 可视化旋转中心
        self.Visualize_force_rotation_center_UI = DebugAxes()  # 可视化旋转中心
        # 机械臂实际执行频率，ur5真实通讯频率是120hz
        self._timeStep = 120
        self.t = (1 / self._timeStep) * 2
        self.action_dim = 7
        # --------------------------- 重置关节至初始状态--------------------------------
        self.init_joint_val = [0.29572491999307804, -1.3445488199311688, -2.1969757544996567,
                               -1.1709496137703805, 1.572643027403216, 0.29166692172730446]
        # 关节跳跃值
        self.joint_damping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

        # 自带数据库地址
        self.urdf_root_path = pybullet_data.getDataPath()

        # admittance controller parameter -- modified by YbZhou
        self.ur5EndEffectorIndex = 7
        # rospy.init_node('admittance_node', anonymous=True)
        # 构建导纳控制的三个系数矩阵
        """
        when k=80, d=40, m=10 => w=5 l=1 : need 13 steps
        when k=80, d=56, m=10 => w=2.82 l=1 : need 17 steps
        """
        self.In_M = 0.1  # M = 10， inverse_M = 1/M
        self.translational_stiffness = 10
        self.rotational_stiffness = 10
        self.translational_damping = 5
        self.translational_damping = 5

        self.Inverse_M = np.mat(self.In_M * np.eye(6))

        self.stiffness = np.mat(np.block([
            [self.translational_stiffness * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), self.rotational_stiffness * np.eye(3)]
        ]))

        self.damping = np.mat(np.block([
            [self.translational_damping * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), self.translational_damping * np.eye(3)]
        ]))
        self.writer = SummaryWriter('./HDQN_peg/collect_dataset_log')

        self.current_pose = np.zeros((3, 1))
        self.current_orie = np.zeros((4, 1))
        self.quat_rot_err = np.zeros(4)
        self.fake_wrench = np.zeros(6)
        self.current_twist_lin = np.zeros((3, 1))
        self.current_twist_ang = np.zeros((3, 1))
        self.arm_desired_twist_ = np.mat(np.zeros((6, 1)))
        self.arm_desired_position_ = np.mat(np.zeros((3, 1)))
        self.arm_max_acc_ = 20.0
        self.first_pose_err = True
        self.period = 0.5
        # self.loop_rate = rospy.Rate(100)
        # try:
        #     self.subscriber_ee_state = rospy.Subscriber(("/cartesian_velocity_controller_sim/ee_state"),
        #                             PoseTwist, self.arm_state_callback, queue_size=1)
        #     self.subscriber_wrench = rospy.Subscriber(("/wrench_fake"),
        #                             WrenchStamped, self.wrench_callback, queue_size=1)
        # except KeyError:
        #     # 如果参数不存在，打印错误信息并退出
        #     rospy.logerr("Couldn't retrieve the topic name for the state of the arm.")
        #     exit(-1)

        # 第一步，连接仿真环境
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
        # 添加桌子模型
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[-1.01, 0, -0.315 - 0.02])

        self.ur5_id = p.loadURDF(
            "/home/ur/YbZhou/compliant_control_real_ws/src/Compliant-Control-and-Application-real-devel/Controllers_Algorithms/control_algorithm/Admittance/scripts/UR5_TCP-main/ur_description/urdf/ur5_robot_sensor_pos2_stand_eelink.urdf",
            basePosition=[0, 0, 0.1], flags=9)
        for j in range(p.getNumJoints(self.ur5_id)):
            # print(j,p.getJointState(self.ur5_id,j))
            print(j, p.getJointInfo(self.ur5_id, j))

    def arm_state_callback(self, msg):
        self.current_pose[0] = msg.pose.position.x
        self.current_pose[1] = msg.pose.position.y
        self.current_pose[2] = msg.pose.position.z

        self.current_orie[0] = msg.pose.orientation.x
        self.current_orie[1] = msg.pose.orientation.x
        self.current_orie[2] = msg.pose.orientation.x
        self.current_orie[3] = msg.pose.orientation.x

        self.current_twist_lin[0] = msg.twist.linear.x
        self.current_twist_lin[1] = msg.twist.linear.y
        self.current_twist_lin[2] = msg.twist.linear.z
        self.current_twist_ang[0] = msg.twist.angular.x
        self.current_twist_ang[1] = msg.twist.angular.y
        self.current_twist_ang[2] = msg.twist.angular.z
        # print("current_pose, current_orie, current_twist_lin, current_twist_ang", self.current_pose, self.current_orie, self.current_twist_lin, self.current_twist_ang)

    def wrench_callback(self, msg):
        self.fake_wrench = msg.wrench.force.x
        self.fake_wrench = msg.wrench.force.y
        self.fake_wrench = msg.wrench.force.z

    # 获取机器人受力
    def getForceTorque(self):
        """
        获取机器人当前关节力/力矩
        """

        data_sub = 3
        FT_o = np.zeros([data_sub, 6])

        for i in range(data_sub):
            FT_o[i, :] = np.array(self.Fusb.get_current_ft(), dtype=float).reshape(1, 6)

        f_T = np.mean(FT_o[1:], axis=0)
        force = f_T[0:3]
        torque = f_T[3:]
        d = 0.042 - 0.0034
        FT = [-int(force[2] * 10) / 10, -int(force[0] * 10) / 10, -int(force[1] * 10) / 10,
              int(torque[2] * 100) / 100, -int((torque[0] - force[1] * d) * 100) / 100,
              -int((torque[1] + d * force[0]) * 100) / 100]
        print(FT)
        return FT

    # 为了和机器人tool关节保持一致，作了变化
    def getForceTorqueTBaselink(self):
        """
        获取机器人当前关节力/力矩, 基坐标是base_link, 末端坐标是tool0
        """

        data_sub = 3
        FT_o = np.zeros([data_sub, 6])

        for i in range(data_sub):
            FT_o[i, :] = np.array(self.Fusb.get_current_ft(), dtype=float).reshape(1, 6)

        f_T = np.mean(FT_o[1:], axis=0)
        force = f_T[0:3]
        torque = f_T[3:]
        d = 0.042 - 0.0034
        FT = [int(force[0] * 10) / 10, int(force[2] * 10) / 10, -int(force[1] * 10) / 10,
              int(torque[2] * 100) / 100, -int((torque[0] - force[1] * d) * 100) / 100,
              -int((torque[1] + d * force[0]) * 100) / 100]
        print(FT)

        return FT

    # 获取机器人纯净受力，不做改变
    def getPureForceTorque(self):
        """
        获取机器人当前关节力/力矩
        """

        data_sub = 3
        FT_o = np.zeros([data_sub, 6])

        for i in range(data_sub):
            FT_o[i, :] = np.array(self.Fusb.get_current_ft(), dtype=float).reshape(1, 6)

        f_T = np.mean(FT_o[1:], axis=0)
        force = f_T[0:3]
        torque = f_T[3:]
        d = 0.042 - 0.0034
        FT = [int(force[0] * 10) / 10, int(force[1] * 10) / 10, int(force[2] * 10) / 10,
              int(torque[2] * 100) / 100, -int((torque[0] - force[1] * d) * 100) / 100,
              -int((torque[1] + d * force[0]) * 100) / 100]
        print(FT)
        return FT

    # 获取机器人纯净受力，不做改变
    def getPureForceTorqueEnd(self):
        """
        获取机器人当前关节力/力矩
        """

        data_sub = 3
        FT_o = np.zeros([data_sub, 6])

        for i in range(data_sub):
            FT_o[i, :] = np.array(self.Fusb.get_current_ft(), dtype=float).reshape(1, 6)

        f_T = np.mean(FT_o[1:], axis=0)
        force = f_T[0:3]
        torque = f_T[3:]
        d = 0.042 - 0.0034
        FT = [int(force[2] * 10) / 10, int(force[0] * 10) / 10, int(force[1] * 10) / 10,
              int(torque[2] * 100) / 100, -int((torque[0] - force[1] * d) * 100) / 100,
              -int((torque[1] + d * force[0]) * 100) / 100]
        print(FT)
        return FT

    # 机器人纯净受力转换到too关节下，力传感器坐标与tool坐标不一致
    def getPureForceTorqueToTool(self):
        """
        获取机器人当前关节力/力矩
        """

        data_sub = 3
        FT_o = np.zeros([data_sub, 6])

        for i in range(data_sub):
            FT_o[i, :] = np.array(self.Fusb.get_current_ft(), dtype=float).reshape(1, 6)

        f_T = np.mean(FT_o[1:], axis=0)
        force = f_T[0:3]
        torque = f_T[3:]
        d = 0.042 - 0.0034
        FT = [int(-force[0] * 10) / 10, -int(force[2] * 10) / 10, -int(force[1] * 10) / 10,
              int(torque[2] * 100) / 100, -int((torque[0] - force[1] * d) * 100) / 100,
              -int((torque[1] + d * force[0]) * 100) / 100]
        # print(FT)
        return FT

    # 定义绕 z 轴旋转 180 度的变换矩阵
    def rotation_z_180(self, degree):
        theta = np.radians(degree)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def get_cart_to_base_force_torque(self, xyz_force, xyz_torque, position, orientation):
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

        # tool 到 ee 的变换矩阵
        T_ee = self.rotation_z_180(0)

        end_orn = Rotation.from_quat(orientation).as_matrix()
        wcT = np.eye(4)
        wcT[:3, :3] = end_orn[:3, :3]
        wcT[:3, 3] = position

        # 计算 tool 到 base 的变换矩阵
        T_base = np.dot(wcT, T_ee)

        # 转换力
        force_base = T_base[:3, :3] @ xyz_force

        # 转换力矩
        r = np.array(position)
        torque_base = T_base[:3, :3] @ xyz_torque

        # 组合结果
        force_torque_base = np.vstack((force_base, torque_base))
        return force_torque_base

    def get_compensate_ft(self):
        """
        力传感器连续零漂处理（导纳控制中使用）
        """
        FT = self.getForceTorque()
        print('get_compensate_ft中的FT：', FT)

        FT = [FT[0] - self.ft_compenstate[0], FT[1] - self.ft_compenstate[1], FT[2] - self.ft_compenstate[2],
              FT[3] - self.ft_compenstate[3], FT[4] - self.ft_compenstate[4], FT[5] - self.ft_compenstate[5]]
        if FT[0] < 0:
            FT[0] = 0.0111
        return FT

    def reset_Force_sensor(self):
        '''传感器重新连接，自动置零'''
        A = True
        time.sleep(2)
        while A == True:
            try:
                self.Fusb.close()
                self.Fusb = Forceusb()
                A = False
            except:
                time.sleep(0.5)
                A = True

    # 位置导纳函数，使用movej
    def admittance_position(self, M_position, B_position, K_position):
        '''位置导纳'''
        # posture_new = p.getLinkState(self.ur5_id, 7)[5]
        m_position = []
        fext = self.getForceTorque()[0:3]  # 传感器端
        # fext = self.Dynamic_rotation_center()[0:3]  # 动态力矩中心
        if np.abs(np.array(fext)).max() < 0.3:
            m_position = [0, 0, 0]
            # self.ft_compenstate = self.getForceTorque()
        else:
            # print("fext", fext)
            for i in range(3):
                xd = self.solution(-fext[i], M_position, B_position[i], K_position)
                m_position.append(xd)

            m_position = [m_position[0], m_position[1], m_position[2]]
        return m_position

    def transform_force_torque_base2end(self, xyz_force, xyz_torque):
        force_torque_end = np.hstack((xyz_force, xyz_torque))
        rotation_ft_base = self.get_rotation_matrix("/base_link", "/tool0")
        wrench_external_ = rotation_ft_base * force_torque_end

        return wrench_external_

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

    def get_rotation_matrix(self, from_frame, to_frame):
        """
        获取从 from_frame 到 to_frame 的旋转矩阵。

        :param from_frame: 源坐标系名称
        :param to_frame: 目标坐标系名称
        :return: 6x6 的旋转矩阵（numpy.ndarray），如果获取失败则返回 None
        """
        listener = tf.TransformListener()
        rotation_matrix = np.zeros((6, 6))

        try:
            # 等待 TF 数据
            listener.waitForTransform(from_frame, to_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = listener.lookupTransform(from_frame, to_frame, rospy.Time(0))

            # 将四元数转换为 3x3 旋转矩阵
            rotation_from_to = Rotation.from_quat(rot).as_matrix()

            # 构建 6x6 的旋转矩阵
            rotation_matrix[:3, :3] = rotation_from_to
            rotation_matrix[3:, 3:] = rotation_from_to

            return rotation_matrix

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as ex:
            rospy.logwarn_throttle(1, f"Waiting for TF from: {from_frame} to: {to_frame}")
            return None

    # position admittance controller, modified by YbZhou
    def position_velocity_controller(self, target_pose, target_orie, Inv_M_position, B_position_, K_position_):
        if (self.first_pose_err == True) and (
                np.abs(np.array(target_pose)) - np.abs(np.array(self.current_pose))).max() > 0.001:
            self.first_pose_err = False
            return False

        Fext = self.getPureForceTorqueToTool()[0:3]
        Text = self.getPureForceTorqueToTool()[3:]

        if np.abs(np.array(Fext)).max() < 1:
            Fext = np.array([0, 0, 0])
        if np.abs(np.array(Text)).max() < 1:
            Text = np.array([0, 0, 0])

        # 坐标转换
        force_torque_base = self.cart_to_base_force_torque(np.array(Fext), np.array(Text), self.current_pose[:, 0],
                                                           self.current_orie[:, 0])
        force_external = np.mat(
            [[force_torque_base[0][0]], [force_torque_base[0][1]], [force_torque_base[0][2]], [0], [0], [0]])

        # Position error
        dx = self.current_pose[0] - target_pose[0]
        dy = self.current_pose[1] - target_pose[1]
        dz = self.current_pose[2] - target_pose[2]
        pose_err = np.mat([[dx[0]], [dy[0]], [dz[0]], [0], [0], [0]])
        if np.abs(np.asarray(pose_err)).max() < 0.0001:
            pose_err = np.mat([[0], [0], [0], [0], [0], [0]])

        # Orientation error
        if np.dot(np.transpose(target_orie), self.current_orie) < 0.0:
            self.current_orie = -self.current_orie

        current_orie_matrix = Rotation.from_quat(self.current_orie[:, 0]).as_matrix()
        target_orie = Rotation.from_quat(target_orie).as_matrix()
        target_orie_inv = target_orie.T
        quat_rot_err_tmp = np.dot(current_orie_matrix, target_orie_inv)
        quat_rot_err_tmp = Rotation.from_matrix(quat_rot_err_tmp).as_quat()

        # quat_rot_err_tmp = self.current_orie * np.linalg.inv(target_orie)
        np.copyto(self.quat_rot_err, quat_rot_err_tmp)

        if np.linalg.norm(self.quat_rot_err) > 1e-3:
            self.quat_rot_err = self.quat_rot_err / np.linalg.norm(self.quat_rot_err)
        axis, angle = tfs.quaternions.quat2axangle(self.quat_rot_err)  # 创建一个四元数
        axangle_err = axis * angle
        pose_err[3][0] = axangle_err[0]
        pose_err[4][0] = axangle_err[1]
        pose_err[5][0] = axangle_err[2]

        coupling_wrench_arm = B_position_ * self.arm_desired_twist_ + K_position_ * pose_err
        arm_desired_accelaration = Inv_M_position * (- coupling_wrench_arm + force_external)
        arm_acc_norm = np.linalg.norm(arm_desired_accelaration[:, :3])
        if (arm_acc_norm > self.arm_max_acc_):
            # print("Admittance generates high arm accelaration!", arm_acc_norm)
            arm_desired_accelaration[:, :3] *= (self.arm_max_acc_ / arm_acc_norm)
        print("pose_err : ", pose_err)

        # 获取预期的循环周期时间
        duration = self.loop_rate.sleep_dur

        # 更新 arm_desired_twist_adm_
        self.arm_desired_twist_ += arm_desired_accelaration * duration.to_sec()
        print("desired_twist 1 : ", self.arm_desired_twist_)

        # return self.arm_desired_twist_

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

    def test_hybrid_controller(self, target_pose, target_orie, real_robot_connect):
        self.apply_hybrid_controller(np.concatenate((target_pose, target_orie), axis=0), real_robot_connect,
                                     physicsClientId=self.physicsClient_use, if_test=True)

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
        desired_force_z = 5  # N
        desired_force_rz = 0.5  # N
        desired_force_xy = 0.0  # N
        desired_force_r = 0.0
        Kp_force = 1  # 比例增益
        Ki_force = 0.2  # 积分增益
        Kp_torque = 2  # 比例增益
        Ki_torque = 0.5  # 积分增益
        Kd_force = 0.01  # 微分增益
        integral_force_error_z = 0.0
        integral_force_error_rz = 0.0
        previous_force_error_z = 0.0
        previous_force_error_rz = 0.0
        delta_z_max = 50
        delta_rz_max = 10
        # 获取当前的关节力和力矩
        FT = self.getPureForceTorqueEnd()

        Fext = FT[0:3]
        Text = FT[3:]
        force_torque_base = self.cart_to_base_force_torque(np.array(Fext), np.array(Text), self.current_pose,
                                                           self.current_orie)
        print(force_torque_base)

        # Text = [0, 0, 0]

        action_to_target_pose[0] = action[0] + self.zero_Position[0]
        action_to_target_pose[1] = action[1] + self.zero_Position[1]
        action_to_target_pose[2] = action[2] + self.zero_Position[2]

        action_to_target_orie[0] = action[3]
        action_to_target_orie[1] = action[4]
        action_to_target_orie[2] = action[5]
        action_to_target_orie[3] = action[6]

        wrench_z = force_torque_base[0][2]
        wrench_rz = force_torque_base[1][2]

        self.force_x = Fext[0]
        self.force_y = Fext[1]
        self.force_z = Fext[2]

        self.Torque_x = Text[0]
        self.Torque_y = Text[1]
        self.Torque_z = Text[2]

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

        self.force_error_rz = wrench_rz - desired_force_rz

        integral_force_error_rz += self.force_error_rz * self.duration

        # 计算力控制输出
        force_control_output_rz = Kp_torque * self.force_error_rz + \
                                  Ki_torque * integral_force_error_rz + \
                                  0

        #  Kd_force * derivative_force_error_z
        force_z_adjustment = force_control_output_z
        force_rz_adjustment = force_control_output_rz
        pos_z_adjustment = max(min(force_z_adjustment, delta_z_max), -delta_z_max)
        pos_z_adjustment = pos_z_adjustment / 10  # 手动补偿

        pos_rz_adjustment = max(min(force_rz_adjustment, delta_rz_max), -delta_rz_max)
        pos_rz_adjustment = pos_rz_adjustment * 50  # 手动补偿

        if wrench_z > 1:
            print("wrench_z : ", wrench_z)
            # pos_z_adjustment = pos_z_adjustment / 20
        for i in range(1):
            force_torque_base[0][:] = [x if abs(x) >= 0.1 else 0.0 for x in force_torque_base[0][:]]
            force_torque_base[1][:] = [x if abs(x) >= 0.1 else 0.0 for x in force_torque_base[1][:]]
            if Fext[1] > 1 or Text[2] > 1:
                desired_force_r = -0.0
                desired_force_xy = -0.0
            force_external = np.mat(
                [[force_torque_base[0][0] - desired_force_xy], [force_torque_base[0][1] - desired_force_xy], [0.0],
                 [force_torque_base[1][0] - desired_force_r], [force_torque_base[1][1] - desired_force_r],
                 [force_torque_base[1][2] - desired_force_r]])
            # force_external = np.mat([[0], [Fext[1]], [0], [0], [0], [0]])

            # 获取当前的关节状态
            self.current_Position = self.current_pose
            self.current_Orientation = self.current_orie
            action_to_current[0] = self.current_Position[0] - action_to_target_pose[0]  # 计算当前位置与目标位置的差值
            action_to_current[1] = self.current_Position[1] - action_to_target_pose[1]
            action_to_current[2] = self.current_Position[2] - action_to_target_pose[2]

            self.orientation_err = np.dot(self.current_Orientation, self.zero_Orientation)
            orien_angle_err = 2 * np.arccos(self.orientation_err)
            self.angle_err = orien_angle_err * 180 / np.pi

            # diff = self.quaternion_to_euler(self.current_Orientation, self.zero_Orientation)
            current_orie_matrix = Rotation.from_quat(self.current_Orientation).as_matrix()
            action_to_target_orie_matrix = Rotation.from_quat(self.zero_Orientation).as_matrix()
            target_orie_inv = action_to_target_orie_matrix.T
            quat_rot_err_tmp = np.dot(current_orie_matrix, target_orie_inv)

            quat_rot_err_ = Rotation.from_matrix(quat_rot_err_tmp).as_rotvec()  # rad
            # quat_rot_err_tmp = quat_rot_err_ / 1 # for test, is related to the size of target pose that is input
            # quat_rot_err_tmp = quat_rot_err_ * 60
            quat_rot_err_tmp = quat_rot_err_ * 1
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
            if abs(action_to_current[0]) < 1e-3:
                action_to_current[0] = 0.0
            if abs(action_to_current[1]) < 1e-3:
                action_to_current[1] = 0.0
            if abs(action_to_current[2]) < 1e-3:
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
            self.pose_err = np.mat([[-self.dx], [-self.dy], [0.0], [-self.rx], [-self.ry], [0.0]])
            self.FT_err = np.mat([[0.0], [0.0], [pos_z_adjustment], [0.0], [0.0], [pos_rz_adjustment]])

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

    def quaternion_to_euler(self, q1, q2):
        # 将四元数转换为欧拉角(ZYX顺序)
        rot1 = Rotation.from_quat(q1)
        rot2 = Rotation.from_quat(q2)
        euler1 = rot1.as_euler('xyz', degrees=True)
        euler2 = rot2.as_euler('xyz', degrees=True)

        # 计算角度差异(处理角度环绕)
        diff = np.abs(euler2 - euler1)
        diff = np.where(diff > 180, 360 - diff, diff)

        return diff

    def apply_hybrid_controller(self, action, real_robot_connect, physicsClientId, if_test=False):
        """ Make a step in simulation """
        position_arrive = False
        """
        k=40,d=28.28,m=10
        """
        self.translational_stiffness = 40
        self.rotational_stiffness = 40
        self.translational_damping = 28.28
        self.translational_damping = 28.28
        self.duration = 0.2
        self.zero_Orientation[0] = action[3]
        self.zero_Orientation[1] = action[4]
        self.zero_Orientation[2] = action[5]
        self.zero_Orientation[3] = action[6]
        self.reset_controller()
        while 1:
            desired_twist, deta_desired_twist = self.impedance_controller(action, physicsClientId)
            desired_twist = np.array(desired_twist)
            print("desired_twist : ", desired_twist)
            # real_robot_connect.speed_j(desired_twist)

            # real_robot_connect.speedj([0.0, 0.0, -0.2, 0.00, 0.0, 0.0])
            joint_velocities = self.send_commands_to_sim_robot(desired_twist[0][0], desired_twist[1][0],
                                                               desired_twist[2][0], desired_twist[3][0],
                                                               desired_twist[4][0], desired_twist[5][0],
                                                               physicsClientId, real_robot_connect)
            real_robot_connect.speedj(
                [joint_velocities[0], joint_velocities[1], joint_velocities[2], joint_velocities[3],
                 joint_velocities[4], joint_velocities[5]], 0.4)
            time.sleep(0.4)
            self.solve_steps = self.solve_steps + 1
            # print("self.force_error_z : ", self.force_error_z)
            joint_positions = p.getJointState(self.ur5_id, 6, physicsClientId=physicsClientId)[0]
            # print("self.position_error_y : ", self.position_error_y)
            self.writer.add_scalars("force_error_z",
                                    {"force_error_z": self.force_error_z}, self.solve_steps)
            self.writer.add_scalars("rz",
                                    {"rz": self.rz}, self.solve_steps)
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
            self.writer.add_scalars("joint_positions",
                                    {"joint_positions": joint_positions}, self.solve_steps)
            # if (self.rz < 5e-6 and self.dz < 5e-6) or self.is_in_range(1, self.orientation_err, 5e-6) or self.angle_err < 0.1 or self.angle_err==None:
            if if_test == True:
                if (abs(self.rz) < 5e-2 and abs(
                        self.deta_dz) < 5e-8) or self.insert_depth > 0.435 or self.solve_steps > 3000:
                    # if (abs(self.rz) < 5e-6):
                    # print("force err success")
                    break
            else:
                if (abs(self.rz_tmp) < 5e-3) or self.insert_depth > 0.435 or self.solve_steps > 1000:
                    # if (abs(self.rz) < 5e-6):
                    # print("force err success")
                    break

    def send_commands_to_sim_robot(self, vx, vy, vz, wx, wy, wz, physicsClientId, real_robot_connect):
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
        joint_positions = real_robot_connect.get_state_joint()
        joint_velocities = [0] * num_joints  # 假设初始关节速度为0

        # 计算雅可比矩阵
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.ur5_id, self.ur5EndEffectorIndex,
                                                                localPosition=com_trn,
                                                                objPositions=list(joint_positions),
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

        return joint_velocities

    # 姿态导纳
    def admittance_posture(self, M_posture, B_posture, K_posture):
        posture = []
        # fext = self.get_compensate_ft()[3:6]  # 传感器端面
        f_t = self.getForceTorque()  # 传感器端
        fext = f_t[:3]
        torue = f_t[3:]
        # print('"var forceture:',fext )
        if np.abs(np.array(fext)).max() < 1:
            posture = [0, 0, 0]
        else:
            for i in range(3):
                xd = self.solution(-torue[i] * 10, M_posture, B_posture, K_posture)
                posture.append(xd)

        posture_x = 0  # posture[0]
        posture_y = posture[1]
        posture_z = posture[2]
        m_posture = [posture_x, posture_y, posture_z]
        return m_posture

    # 导纳求解
    def solution(self, fext, M, B, K):

        xd = (B * fext / (2 * K * sqrt(B ** 2 - 4 * K * M)) - fext / (2 * K)) * exp(
            self.t * (-B - sqrt(B ** 2 - 4 * K * M)) / (2 * M)) + (
                     -B * fext / (2 * K * sqrt(B ** 2 - 4 * K * M)) - fext / (2 * K)) * exp(
            self.t * (-B + sqrt(B ** 2 - 4 * K * M)) / (2 * M)) + fext / K
        return complex(xd).real

    def go_speed(self, target_pos, target_orie):
        # ------------------------------------------求解器-------------------------------------------------------
        self.target_robot_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_id,
            endEffectorLinkIndex=7,
            targetPosition=target_pos,
            targetOrientation=target_orie,
            jointDamping=self.joint_damping, )

        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.target_robot_joint_angles[i])
        p.stepSimulation()

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.target_robot_joint_angles),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(25):
            p.stepSimulation()

        # robot_jiont_states = np.zeros([6])
        # for j in [1, 2, 3, 4, 5, 6]:
        #     robot_jiont_states[j - 1] = p.getJointState(self.ur5_id, j)[0]
        '''连续获取真实机器人关节值有误，谨慎使用'''
        robot_jiont_states = np.array(self.real_robot_connect.UR_30003rt('q target')[1])
        speed_jiont = self.target_robot_joint_angles - robot_jiont_states
        self.real_robot_connect.speedj(speed_jiont)
        time.sleep(2.01)  # 等待时间略大于执行时间0.01s,如果想做到顺滑，需要轨迹插值，且等待时间小于执行时间，或者使用速度控制器（内部自己做了插值）

    def go(self, target_pos, target_orie):
        # ------------------------------------------求解器-------------------------------------------------------
        self.target_robot_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_id,
            endEffectorLinkIndex=7,
            targetPosition=target_pos,
            targetOrientation=target_orie,
            jointDamping=self.joint_damping, )

        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=self.target_robot_joint_angles[i])
        p.stepSimulation()

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(self.target_robot_joint_angles),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(25):
            p.stepSimulation()
        self.real_robot_connect.movej(self.target_robot_joint_angles)

        time.sleep(2.01)  # 等待时间略大于执行时间0.01s,如果想做到顺滑，需要轨迹插值，且等待时间小于执行时间，或者使用速度控制器（内部自己做了插值）

    def go_speed_l(self, target_pos):

        self.real_robot_connect.speedl(target_pos)
        # time.sleep(2.01)#等待时间略大于执行时间0.01s,如果想做到顺滑，需要轨迹插值，且等待时间小于执行时间，或者使用速度控制器（内部自己做了插值）

    def reset(self):
        self.zero_Position = np.zeros(3)
        self.zero_Orientation = np.zeros(4)
        self.solve_steps = 0

        # 安全设置
        # safe_pose= [0.29480578842411714, -1.2777899899535896, -2.002758482598918, -1.4319075968715884, 1.5708793527071245, 0.2942708693688756]
        # safe_pose= [0.1859, -1.5894, -2.1410, -1.0315, 1.5653, 0.2955] # 测试 force/position controller
        # safe_pose = [0.1468, -1.5007, -1.8448, -1.4533, 1.5505, 1.9787]
        # safe_pose = [0.4128, -1.7264, -2.1350, -0.8508, 1.5708, 0.4098] # tight nut bolt 1
        safe_pose = [0.3479, -1.8175, -2.1000, -0.8219, 1.5708, 1.9953]  # tight nut bolt 2
        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=safe_pose[i])

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(safe_pose),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(240):
            p.stepSimulation()
        self.real_robot_connect.movej(safe_pose)
        time.sleep(2.1)

        self.reset_Force_sensor()

        joint_ = self.real_robot_connect.get_state_joint()
        print("joint: ", joint_)

        self.set_sim_pose(joint_)
        pose = p.getLinkState(self.ur5_id, 8)[4]
        init_orien = [0.0, np.pi / 2, 0.0]  # 垂直向下

        obs = self.get_observation()
        return obs

    def step(self, action):
        n_steps = 1
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
                self.real_robot_connect.speedj(self.target_joint)

        observation = self.get_observation()
        reward = 1
        done = False
        info = None

        return observation, reward, done, info

    def get_observation(self):
        actions = np.zeros((self.action_dim))

        images = self.rscamera.get_data()
        images_hand = self.rscamera.get_data()
        joint_ = self.real_robot_connect.get_state_joint()
        self.set_sim_pose(joint_)
        robot_position = p.getLinkState(self.ur5_id, 7)[4]
        robot_orientation = p.getLinkState(self.ur5_id, 7)[5]

        self.peg_position = np.array(robot_position)
        self.peg_orientation = np.array(robot_orientation)

        action_position = np.array(self.peg_position)
        action_orientation = np.array(self.peg_orientation)
        actions[0:3] = action_position
        actions[3:self.action_dim] = action_orientation
        obs = {
            'agent_pos': actions,
            'image': images,
            'images_hand': images_hand,
        }

        return obs

    def set_sim_pose(self, joint_pose):
        for i in range(6):
            p.resetJointState(bodyUniqueId=self.ur5_id, jointIndex=i + 1, targetValue=joint_pose[i])

        p.setJointMotorControlArray(self.ur5_id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(joint_pose),
                                    forces=np.array([87.0, 87.0, 87.0, 87.0, 60, 60]))

        # ----------------------将更新频率设置为真实频率---------------------
        p.setTimeStep(1.0 / self._timeStep)
        for _ in range(240):
            p.stepSimulation()
