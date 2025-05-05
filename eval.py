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
    env = hydra.utils.instantiate(cfg.task.env, cfg.task.shape_meta)
    device = 'cuda:0'
    device = torch.device(device)
    policy.load_state_dict(torch.load("model_380.pth", map_location=device, pickle_module=dill))
    policy.to(device)
    policy.eval()

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    horizon = cfg.horizon

    obs = env.reset()
    images = np.zeros((1, horizon, *np.transpose(obs['image'], (2,0,1)).shape))
    images_hand = np.zeros((1, horizon, *np.transpose(obs['image_hand'], (2,0,1)).shape))
    depths = np.zeros((1, horizon, *np.transpose(obs['depth'], (2,0,1)).shape))
    agent_pos = np.zeros((1, horizon, *obs['agent_pos'].shape))
    for i in range(horizon):
        images[0, i, ...] = np.transpose(obs['image'], (2,0,1))
        images_hand[0, i, ...] = np.transpose(obs['image_hand'], (2,0,1))
        depths[0, i, ...] = np.transpose(obs['depth'], (2,0,1))
        agent_pos[0, i, ...] = obs['agent_pos']
    observation = {}
    observation['images'] = images
    observation['images_hand'] = images_hand
    observation['depths'] = depths
    observation['agent_pos'] = agent_pos

    done = False
    step_num = 0

    while not done:
        step_start = time.time()

        if step_num % n_action_steps == 0:
            observation['images'][0, 0:n_obs_steps - 1, ...] = observation['images'][0, 1:n_obs_steps, ...]
            observation['images_hand'][0, 0:n_obs_steps - 1, ...] = observation['images_hand'][0, 1:n_obs_steps, ...]
            # observation['depths'][0, 0:n_obs_steps - 1, ...] = observation['depths'][0, 1:n_obs_steps, ...]
            observation['agent_pos'][0, 0:n_obs_steps - 1, ...] = observation['agent_pos'][0, 1:n_obs_steps, ...]

            observation['images'][0, n_obs_steps - 1, ...] = np.transpose(obs['image'], (2,0,1))
            observation['images_hand'][0, n_obs_steps - 1, ...] = np.transpose(obs['image_hand'], (2,0,1))
            # observation['depths'][0, n_obs_steps - 1, ...] = np.transpose(obs['depth'], (2,0,1))
            observation['agent_pos'][0, n_obs_steps - 1, ...] = obs['agent_pos']

            """ visualize point cloud """
            point_cloud = obs['point_cloud']
            # sampled_points = env.uniform_sampling(point_cloud, 0.8)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(sampled_points[:, 3:] / 255.0)
            # o3d.visualization.draw_geometries([pcd])
            """ visualize point cloud """

            actions = policy.predict_action(observation)['action'].cpu().detach().numpy()
            print("actions : ", actions)
        action = actions[0, step_num % n_action_steps, :]
        obs, reward, done, info = env.step(action)

        step_num += 1
        time_until_next_step = 1/env._timeStep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    p.disconnect()


if __name__ == '__main__':
    main()
