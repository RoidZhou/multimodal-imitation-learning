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


def write_zarr(filename, states, actions, episode_ends):
    root = zarr.open(store=filename, mode='w')
    data_group = root.create_group('data')
    data_group.create_dataset('state', shape=states.shape, dtype=states.dtype,
                              chunks=(episode_ends[0], states.shape[1]))
    data_group.create_dataset('action', shape=actions.shape, dtype=actions.dtype,
                              chunks=(episode_ends[0], actions.shape[1]))
    data_group['state'][:] = states
    data_group['action'][:] = actions

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
    env = hydra.utils.instantiate(cfg.task.env)

    num = 100

    states = np.array([])
    actions = np.array([])
    episode_ends = []

    for i in range(num):
        env.reset()
        data = env.collect_force_dataset_delta_orien_improv(20)
        if i == 0:
            states = data['states']
            actions = data['actions']
        else:
            states = np.vstack((states, data['states']))
            actions = np.vstack((actions, data['actions']))
        episode_ends.append(states.shape[0])

    filename = './data/ur5_assembly/ur5_assembly_force.zarr'
    write_zarr(filename, states, actions, episode_ends)


if __name__ == '__main__':
    main()
