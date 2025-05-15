# matplotlib.style.use('ggplot')
import torch.optim as optim
import sys

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
from stable_baselines3 import PPO
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
    agent = hydra.utils.instantiate(cfg.agent, optimizer_spec=optimizer_spec)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(torch.load(cfg.model_path.path, map_location=agent.device, pickle_module=dill))
    policy.to(agent.device)
    policy.eval()

    env = hydra.utils.instantiate(cfg.env, cfg.shape_meta)

    env.reset()

    agent, stats, visits = hdqn_learning(
        env=env,
        agent=agent,
        policy = policy,
        num_episodes=NUM_EPISODES,
        exploration_schedule=exploration_schedule,
        gamma=GAMMA,
    )

    plot_episode_stats(stats)

if __name__ == '__main__':
    main()
