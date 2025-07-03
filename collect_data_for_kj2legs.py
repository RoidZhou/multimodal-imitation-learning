import sys

sys.path.append('./imitation_learning_idp3')

import os
import numpy as np

import pathlib
from omegaconf import OmegaConf
import hydra
import zarr

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

OmegaConf.register_new_resolver("eval", eval, replace=True)


def write_zarr(filename, images, images_hand, depths, states, actions, forces, episode_ends):
    root = zarr.open(store=filename, mode='w')
    data_group = root.create_group('data')
    data_group.create_dataset('image', shape=images.shape, dtype=images.dtype,
                              chunks=(episode_ends[0], images.shape[1], images.shape[2]))
    data_group.create_dataset('image_hand', shape=images_hand.shape, dtype=images_hand.dtype,
                              chunks=(episode_ends[0], images_hand.shape[1], images_hand.shape[2]))
    data_group.create_dataset('depth', shape=depths.shape, dtype=depths.dtype,
                              chunks=(episode_ends[0], depths.shape[1], depths.shape[2]))
    data_group.create_dataset('state', shape=states.shape, dtype=states.dtype,
                              chunks=(episode_ends[0], states.shape[1]))
    data_group.create_dataset('action', shape=actions.shape, dtype=actions.dtype,
                              chunks=(episode_ends[0], actions.shape[1]))
    data_group.create_dataset('force', shape=forces.shape, dtype=forces.dtype,
                              chunks=(episode_ends[0], forces.shape[1]))
    data_group['image'][:] = images
    data_group['image_hand'][:] = images_hand
    data_group['depth'][:] = depths
    data_group['state'][:] = states
    data_group['action'][:] = actions
    data_group['force'][:] = forces

    meta_group = root.create_group('meta')
    meta_group.create_dataset('episode_ends', shape=(len(episode_ends),), dtype=np.int64, chunks=(len(episode_ends),))
    meta_group['episode_ends'][:] = episode_ends


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'imitation_learning_idp3', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    env = hydra.utils.instantiate(cfg.task.env, cfg.task.shape_meta)

    num = 100

    point_clouds = np.array([])
    images = np.array([])
    images_hand = np.array([])
    depths = np.array([])
    states = np.array([])
    actions = np.array([])
    forces = np.array([])
    episode_ends = []

    for i in range(num):
        env.reset()
        data = env.run()
        print("run step : ", i)
        if i == 0:
            # images = data['images']
            # images_hand = data['images_hand']
            # depths = data['depths']
            states = data['states']
            actions = data['states']
            # forces = data['forces']
        else:
            # images = np.vstack((images, data['images']))
            # images_hand = np.vstack((images_hand, data['images_hand']))
            # depths = np.vstack((depths, data['depths']))
            states = np.vstack((states, data['states']))
            actions = np.vstack((actions, data['states']))
            # forces = np.vstack((forces, data['forces']))
        # episode_ends.append(states.shape[0])

    filename = './data/ur5_assembly/ur5_assembly_kjrobot_2legs.zarr'
    write_zarr(filename, states, actions)


if __name__ == '__main__':
    main()
