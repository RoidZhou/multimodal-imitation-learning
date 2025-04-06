import sys

import numpy as np

sys.path.append('../../../imitation_learning_idp3')

import time
import spatialmath as sm
import open3d as o3d
import mujoco
import mujoco.viewer

from imitation_learning_idp3.arm.robot import Robot, UR5e
from imitation_learning_idp3.arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from imitation_learning_idp3.utils import mj


class UR5Grasp3dEnv:

    def __init__(self):
        self.sim_hz = 500
        self.control_hz = 25
        self.latest_action = None
        self.render_cache = None

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.ur5e_joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()
        self.obj_t = np.zeros(3)

        self.step_num = 0
        self.mj_renderer: mujoco.Renderer = None
        self.mj_renderer_depth: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

    def reset(self):
        self.mj_model = mujoco.MjModel.from_xml_path(
            '/home/zhou/autolab/imitation_learning_idp3/imitation_learning_idp3/assets/scenes/scene.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        self.robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self.robot.set_joint(self.robot_q)
        self.ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                 "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.ur5e_joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.robot_q))
        self.robot.set_tool(np.array([0.0, 0.0, 0.15]))
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        px = np.random.uniform(low=1.4, high=1.5)
        py = np.random.uniform(low=0.3, high=0.9)
        pz = 0.83
        T_Box = sm.SE3.Trans(px, py, pz)
        mj.set_free_joint_pose(self.mj_model, self.mj_data, "Box", T_Box)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.obj_t = mj.get_body_pose(self.mj_model, self.mj_data, "Box").t

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer_depth = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.enable_depth_rendering()
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                Ti = sm.SE3.Trans(action[0], action[1], action[2]) * sm.SE3(sm.SO3(self.T0.R))
                self.robot.move_cartesian(Ti)
                joint_position = self.robot.get_joint()
                self.mj_data.ctrl[:6] = joint_position
                mujoco.mj_step(self.mj_model, self.mj_data)

        observation = self._get_obs()
        self.obj_t = mj.get_body_pose(self.mj_model, self.mj_data, "Box").t
        distance = np.linalg.norm(self.robot_T.t[:2] - self.obj_t[:2])
        reward = 1 / (distance + 1)
        done = False
        if distance < 0.01 and np.abs(self.robot_T.t[2] - (self.obj_t[2] + 0.06)) < 0.005:
            done = True

        info = self._get_info()

        self.step_num += 1
        if self.step_num > 10000:
            done = True

        return observation, reward, done, info

    def render(self, mode):
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_renderer_depth is not None:
            self.mj_renderer_depth.close()

    def seed(self, seed=None):
        pass

    def run(self):
        time0 = 0.001
        T0 = self.robot.get_cartesian()
        t0 = T0.t
        R0 = sm.SO3(T0.R)
        t1 = t0.copy()
        R1 = R0.copy()
        planner0 = self.cal_planner(t0, R0, t1, R1, time0)

        time1 = 2.0
        t2 = t1.copy()
        t2[:] = self.obj_t
        t2[2] += 0.15
        R2 = R1.copy()
        planner1 = self.cal_planner(t1, R1, t2, R2, time1)

        time2 = 2.0
        t3 = t2.copy()
        t3[2] = t2[2] - 0.09
        R3 = R2.copy()
        planner2 = self.cal_planner(t2, R2, t3, R3, time2)

        time3 = 0.5
        t4 = t3.copy()
        R4 = R3.copy()
        planner3 = self.cal_planner(t3, R3, t4, R4, time3)

        time_array = np.array([0, time0, time1, time2, time3])
        planner_array = [planner0, planner1, planner2, planner3]
        total_time = np.sum(time_array)

        time_step_num = round(total_time / self.mj_model.opt.timestep) + 1
        every_step_num = 20
        every_epoch_num = time_step_num // every_step_num
        times = np.linspace(0, total_time, time_step_num)
        desired_poses = np.zeros((time_step_num, self.robot.dof))

        states = np.zeros((every_epoch_num, 3))
        actions = np.zeros((every_epoch_num, 3))
        point_clouds = np.zeros((every_epoch_num, self.num_points, 6))

        time_cumsum = np.cumsum(time_array)
        joint_position = self.robot_q.copy()
        for i, timei in enumerate(times):
            for j in range(len(time_cumsum)):
                if timei < time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    self.robot.move_cartesian(planner_interpolate)
                    joint_position = self.robot.get_joint()
                    break

            desired_poses[i, :] = joint_position
        data_num = 0
        time_num = 0
        while self.mj_viewer.is_running():
            step_start = time.time()

            if time_num % every_step_num == 0:

                self.mj_renderer.update_scene(self.mj_data, 0)
                self.mj_renderer_depth.update_scene(self.mj_data, 0)
                img = self.mj_renderer.render()
                depth = self.mj_renderer_depth.render()

                point_cloud = np.zeros((self.height * self.width, 6))
                for h in range(self.height):
                    for w in range(self.width):
                        point_cloud[h * self.width + w, :3] = self.camera_matrix_inv @ np.array(
                            [w * 1.0, h * 1.0, 1.0]) * depth[h, w]
                        point_cloud[h * self.width + w, 3:] = img[h, w, :]

                joint_state = np.array(
                    [mj.get_joint_q(self.mj_model, self.mj_data, name)[0] for name in self.ur5e_joint_names])
                state = self.robot.fkine(joint_state).t
                action = self.robot.fkine(desired_poses[time_num, :]).t

                states[data_num, ...] = state
                actions[data_num, ...] = action
                point_clouds[data_num, ...] = self.uniform_sampling(point_cloud)
                data_num += 1

            time_num += 1
            if time_num >= time_step_num - every_step_num:
                break

            self.mj_data.ctrl[:6] = desired_poses[time_num, :]
            mujoco.mj_step(self.mj_model, self.mj_data)

            self.mj_viewer.sync()
            time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.mj_viewer.close()
        self.mj_renderer.close()
        self.mj_renderer_depth.close()
        return {
            'states': states,
            'actions': actions,
            'point_clouds': point_clouds
        }

    def cal_planner(self, t0, R0, t1, R1, time):
        position_parameter = LinePositionParameter(t0, t1)
        attitude_parameter = OneAttitudeParameter(R0, R1)
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        trajectory_planner = TrajectoryPlanner(trajectory_parameter)
        return trajectory_planner

    def uniform_sampling(self, point_cloud):

        condition = point_cloud[:, 2] < 3.0
        filtered_points = point_cloud[condition, :]
        indices = np.random.permutation(filtered_points.shape[0])[:self.num_points]
        sampled_points = filtered_points[indices, :]
        return sampled_points

    def _get_obs(self):
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        img = self.mj_renderer.render()
        depth = self.mj_renderer_depth.render()

        point_cloud = np.zeros((self.height * self.width, 6))
        for h in range(self.height):
            for w in range(self.width):
                point_cloud[h * self.width + w, :3] = self.camera_matrix_inv @ np.array([w * 1.0, h * 1.0, 1.0]) * \
                                                      depth[h, w]
                point_cloud[h * self.width + w, 3:] = img[h, w, :]
        sampled_points = self.uniform_sampling(point_cloud)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(sampled_points[:, 3:] / 255.0)
        # o3d.visualization.draw_geometries([pcd])

        for i in range(len(self.ur5e_joint_names)):
            self.robot_q[i] = mj.get_joint_q(self.mj_model, self.mj_data, self.ur5e_joint_names[i])
        self.robot_T = self.robot.fkine(self.robot_q)
        agent_pos = self.robot.fkine(self.robot_q).t
        obs = {
            'agent_pos': agent_pos,
            'point_cloud': sampled_points
        }
        self.render_cache = img
        return obs

    def _get_info(self):
        info = {
            'agent_pos': self.robot_T.t[:2],
            'goal_pose': self.obj_t
        }
        return info


if __name__ == '__main__':
    env = UR5Grasp3dEnv()
    env.reset()
    data = env.run()
