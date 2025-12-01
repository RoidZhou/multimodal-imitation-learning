# matplotlib.style.use('ggplot')
import torch.optim as optim
import sys
import glob
sys.path.append('./imitation_learning_idp3')
from imitation_learning_idp3.env.hdqn_env.mdp import UR5Env
from diffusion_policy_3d.policy.hdqn_mdp import OptimizerSpec
from hdqn import hdqn_learning
from utils.plotting import plot_episode_stats
from utils.schedule import LinearSchedule
import pathlib
from omegaconf import OmegaConf
import hydra
import os
import dill
import torch
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer, PPO, A2C
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


NUM_EPISODES = 12000
BATCH_SIZE = 128
GAMMA = 1.0
REPLAY_MEMORY_SIZE = 1000000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

optimizer_spec = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
)

def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init

exploration_schedule = LinearSchedule(2000, 0.1, 1)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

OmegaConf.register_new_resolver("hdqn", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'imitation_learning_idp3', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(torch.load(cfg.model_path.path, map_location=cfg.agent.device, pickle_module=dill))
    policy.to(cfg.agent.device)
    policy.eval()

    agent = hydra.utils.instantiate(cfg.agent, meta_policy=policy, optimizer_spec=optimizer_spec)
    # env = hydra.utils.instantiate(cfg.env, cfg.shape_meta, agent)

    if cfg.n_envs == 1:
        env = hydra.utils.instantiate(cfg.env, cfg.shape_meta, agent)

    else:
        env = make_vec_env(lambda: hydra.utils.instantiate(cfg.env, cfg.shape_meta, agent),
                           cfg.n_envs)
        # env = DummyVecEnv([lambda: hydra.utils.instantiate(cfg.env, cfg.shape_meta, agent)]*cfg.n_envs)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[126, 256], qf=[126, 256]),
    )

    model = PPO(policy="MlpPolicy", env=env,
                batch_size=128, n_steps=64,
                policy_kwargs=policy_kwargs, verbose=1,
                tensorboard_log="./HRL_Logs/")
    model.learn(total_timesteps=cfg.hrl_timesteps)
    if not os.path.exists(f'models/rl_model_{cfg.hrl_timesteps}_steps.zip'):
        model.save(path=f'models/rl_model_{cfg.hrl_timesteps}_steps.zip')
    print("Done training model.")

    # Inference the model
    """
    print("Inference model policy with rendering...")
    saves = glob.glob(f'models/rl_model_{cfg.hrl_timesteps}_steps.zip')
    latest_save = sorted(saves, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
    model.load(path=latest_save)
    obs = env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action_code)

        if done:
            print("Episode done, resetting.")
            obs = env.reset()
    print("Done")
    """
if __name__ == '__main__':
    main()
