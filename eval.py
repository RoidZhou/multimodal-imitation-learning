import sys

import pylab as p

sys.path.append('./imitation_learning_idp3')

import os
import dill
import torch
import time
import numpy as np
import open3d as o3d
import pathlib
from omegaconf import OmegaConf
import hydra
import pybullet as p
from scipy.spatial.transform import Rotation as R

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'imitation_learning_idp3', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy)
    env = hydra.utils.instantiate(cfg.task.env)
    device = 'cuda:0'
    device = torch.device(device)
    policy.load_state_dict(torch.load("model_600.pth", map_location=device, pickle_module=dill))
    policy.to(device)
    policy.eval()

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    horizon = cfg.horizon

    obs = env.reset()
    # 初始化参数
    target_angles = np.array([90, 90, -90])  # 目标姿态
    hole_position = p.getBasePositionAndOrientation(env.tool_id[0])[0]
    hole_pose = np.array(hole_position)
    hole_pose[2] += 0.06

    hole_orientation = R.from_euler('xyz', [90, 90, -90], degrees=True).as_quat()  # 默认固定孔的姿态
    hole_orie = np.array(hole_orientation)
    # 移动到初始位姿
    env.go(hole_pose, hole_orie)

    # 移动到初始插接位姿
    hole_pose[2] = hole_pose[2] - 0.015  # 孔深度0.08
    env.go(hole_pose, hole_orie)

    init_x_angle_err= np.random.uniform(-5, 5)
    init_y_angle_err= np.random.uniform(-5, 5)

    hole_orientation = R.from_euler('xyz', [90+init_x_angle_err, 90+init_y_angle_err, -90], degrees=True).as_quat()  # 默认固定孔的姿态
    hole_orie = np.array(hole_orientation)
    # 移动到初始位姿
    env.go(hole_pose, hole_orie)

    FT = env.getForceTorque()
    Fext = [FT[0], FT[1], FT[2]]
    print("force : ", Fext)
    # point_cloud = np.zeros((1, horizon, *obs['point_cloud'].shape))
    agent_pos = np.zeros((1, horizon, *obs['agent_pos'].shape))
    for i in range(horizon):
        agent_pos[0, i, ...] = obs['agent_pos']
    observation = {}
    observation['agent_pos'] = agent_pos

    done = False
    step_num = 0

    while not done:
        step_start = time.time()

        if step_num % n_action_steps == 0:
            observation['agent_pos'][0, 0:n_obs_steps - 1, ...] = observation['agent_pos'][0, 1:n_obs_steps, ...]

            observation['agent_pos'][0, n_obs_steps - 1, ...] = obs['agent_pos']

            """ visualize point cloud """
            # point_cloud = obs['point_cloud']
            # sampled_points = env.uniform_sampling(point_cloud)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(sampled_points[:, 3:] / 255.0)
            # o3d.visualization.draw_geometries([pcd])
            """ visualize point cloud """

            actions = policy.predict_action(observation)['action'].cpu().detach().numpy()

        action = actions[0, step_num % n_action_steps, :]
        obs, reward, done, info = env.step(action)

        step_num += 1
        time_until_next_step = 1/env._timeStep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    p.disconnect()


if __name__ == '__main__':
    main()
