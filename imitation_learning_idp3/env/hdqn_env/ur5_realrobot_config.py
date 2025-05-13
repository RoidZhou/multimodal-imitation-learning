from math import *
from socket import *
import struct
import serial
import serial.tools.list_ports
from socket import *
import minimalmodbus as mm
import numpy as np


class Forceusb():
    def __init__(self):
        self.BAUDRATE=19200
        self.BYTESIZE=8
        self.PARITY="N"
        self.STOPBITS=1
        self.TIMEOUT=0.2
        self.PORTNAME=self.serial_ports()
        self.SLAVEADDRESS=9
        self.ser=serial.Serial(port=self.PORTNAME, baudrate=self.BAUDRATE, bytesize=self.BYTESIZE, parity=self.PARITY, stopbits=self.STOPBITS, timeout=self.TIMEOUT)
        self.packet = bytearray()
        self.sendCount=0
        while self.sendCount<50:
            self.packet.append(0xff)
            self.sendCount=self.sendCount+1
        self.ser.write(self.packet)
        self.ser.close()
        #Communication setup
        mm.BAUDRATE=self.BAUDRATE
        mm.BYTESIZE=self.BYTESIZE
        mm.PARITY=self.PARITY
        mm.STOPBITS=self.STOPBITS
        mm.TIMEOUT=self.TIMEOUT
        self.ft300=mm.Instrument(self.PORTNAME, slaveaddress=self.SLAVEADDRESS)
        self.registers=self.ft300.read_registers(180,6)
        # Save measured values at rest. Those values are use to make the zero of the sensor.
        self.fxZero=self.forceConverter(self.registers[0])
        self.fyZero=self.forceConverter(self.registers[1])
        self.fzZero=self.forceConverter(self.registers[2])
        self.txZero=self.torqueConverter(self.registers[3])
        self.tyZero=self.torqueConverter(self.registers[4])
        self.tzZero=self.torqueConverter(self.registers[5])

        self.ft_data = []
        self.num = 0

    def close(self):
        self.ft300.serial.close()

    def serial_ports(self):#自动寻找端口
        ports = list(serial.tools.list_ports.comports())
        for port_no, description, address in ports:
            if 'USB' in description:
                return port_no


    def forceConverter(self,forceRegisterValue):
        """Return the force corresponding to force register value.

        input:
            forceRegisterValue: Value of the force register

        output:
            force: force corresponding to force register value in N
        """
        force=0
        forceRegisterBin=bin(forceRegisterValue)[2:]
        forceRegisterBin="0"*(16-len(forceRegisterBin))+forceRegisterBin
        if forceRegisterBin[0]=="1":
            #negative force
            force=-1*(int("1111111111111111",2)-int(forceRegisterBin,2)+1)/100
        else:
            #positive force
            force=int(forceRegisterBin,2)/100
        return force


    def torqueConverter(self,torqueRegisterValue):
        """Return the torque corresponding to torque register value.

        input:
            torqueRegisterValue: Value of the torque register

        output:
            torque: torque corresponding to force register value in N.m
        """
        torque=0

        torqueRegisterBin=bin(torqueRegisterValue)[2:]
        torqueRegisterBin="0"*(16-len(torqueRegisterBin))+torqueRegisterBin
        if torqueRegisterBin[0]=="1":
            #negative force
            # torque=-1*(int("1111111111111111",2)-int(torqueRegisterBin[1:],2)+1)/1000
            torque=-1*(int("1111111111111111",2)-int(torqueRegisterBin,2)+1)/1000
        else:
            #positive force
            torque=int(torqueRegisterBin,2)/1000
        return torque

    def SN(self,snValue):
        pass


    def get_current_ft(self):
        """
        获得力传感器数值
        """
        registers=self.ft300.read_registers(180,6)
        fx=round(self.forceConverter(registers[0])-self.fxZero,2)
        fy=round(self.forceConverter(registers[1])-self.fyZero,2)
        fz=round(self.forceConverter(registers[2])-self.fzZero,2)
        tx=round(self.torqueConverter(registers[3])-self.txZero,2)
        ty=round(self.torqueConverter(registers[4])-self.tyZero,2)
        tz=round(self.torqueConverter(registers[5])-self.tzZero,2)
        ft = [fx, fy, fz, tx, ty, tz]
        self.ft_data.append(ft)
        # print("ft", ft[3:6])
        return ft

# Fusb=Forceusb()
# force=Fusb.get_current_ft()
# print(force)

class socket_TCp_UR30003():
    def __init__(self):
        self.host_name = "192.168.1.102"
        self.port_num = 30003
        self.ClientSocket = socket(AF_INET,SOCK_STREAM)
        self.ClientSocket.connect((self.host_name,self.port_num))

    def UR_30003Script(self, send_data):
        # print(send_data)
        self.ClientSocket.send(send_data.encode('utf8'))


    def UR_30003rt(self,Meaning):

        dic= {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d','I target': '6d',
            'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
            'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
            'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
            'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d', 'Tool Accelerometer values': '3d',
            'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd', 'V main': 'd',
            'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd', 'Elbow position': '3d', 'Elbow velocity': '3d'}
        data=self.ClientSocket.recv(1220)
        ii=range(len(dic))
        for key,i in zip(dic,ii):
            fmtsize=struct.calcsize(dic[key])
            info,data=data[0:fmtsize],data[fmtsize:]
            fmt="!"+dic[key]
            dic[key]=dic[key],struct.unpack(fmt, info)
        f=1

        return dic[Meaning]



    def movej_offset(self,offset):
        '''TCP_pos:是当前tool的'''

        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        global pose=get_actual_tcp_pose()
        global P= pose_trans(pose,p[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}])
        # global P2=get_inverse_kin(P)
        movej(P, a=0.05, v=0.25, t=0, r=0)
        

    end
        '''

        self.UR_30003Script(send_data)#30003发送


    def movej(self,offset):
        '''TCP_pos:是当前tool的'''

        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        # global pose=[{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}]
        # popup(pose)
        movej([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], a=0.05, v=0.25, t=1.5, r=0)
       

    end
            '''
        self.UR_30003Script(send_data)#30003发送
       
       
 #直接控制六个轴的速度
    def speedj(self,offset):
        send_data = f'''
    def whf():
        set_tcp(p[0,0,0,0,0,0])
        speedj([{offset[0]},{offset[1]},{offset[2]},{offset[3]},{offset[4]},{offset[5]}], 0.2,0.5)
    end
            '''
        self.UR_30003Script(send_data)#30003发送
        
#控制末端速度类似speedl，但是是欧拉角下  
    def speedj_offset(self,offset):

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
        self.UR_30003Script(send_data)#30003发送

        # [-0.08972922650022963, -1.6219628747300388, -2.0232094758251753, -1.0464530101386091, 1.5717372302313297,
        #  -0.08981790491708695]





R=socket_TCp_UR30003()
# # jionts=R.UR_30003rt('q target')
# # print(jionts)
# # import numpy as np
# # import math
# # np.multiply(np.array(jionts[1]),180.0000000/math.pi)
# R.movej_offset([-0.01,0,-0,-0,0,0])#写在类中global pose=get_actual_tcp_pose()不执行，所以与offest相关的函数必须写在类外，原因不清楚
# # R.movej([-0.08972922650022963, -1.6219628747300388, -2.0232094758251753, -1.0464530101386091, 1.5717372302313297,-0.08981790491708695])
#
# #test speedj
# # R.speedj_offset([0.0, 0.03,0,0.002,0,0])
# # R.speedj([0.0, 0.0,0,0.00,0,0.0])
#
# print('dd')




'''robot jiont states rad to  *180/np.pi'''
# print(np.array([0.29470252959723864, -1.3434761825351185, -2.1951895338764267, -1.1738231388222164, 1.5726177712823106, 0.2906822850770858])*180/np.pi)

'''实验记录：pybullet姿态下对中的时的关节角度：'''
#正好与方块表面贴合 (0.29423205642086486, -1.3442299295978728, -2.1957716121972233, -1.1724944331805487, 1.5726282071398505, 0.29019361321914783)
#从方块表面保持安全距离 (0.294092870401073, -1.33090764529892, -2.171405356667104, -1.210179931805639, 1.572316109553702, 0.29055288564098003)







    # print('demo test')