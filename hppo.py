import numpy as np
from collections import defaultdict
from itertools import count
import random
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory
from utils import plotting

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def one_hot_goal(goal):
    vector = np.zeros(6)
    vector[goal-1] = 1.0
    return np.expand_dims(vector, axis=0)

def hppo_learning(
    env,
    agent,
    policy,
    num_episodes,
    exploration_schedule,
    gamma=1.0,
    ):

    """The h-DQN learning algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    agent:
        a h-DQN agent consists of a meta-controller and controller.
    num_episodes:
        Number (can be divided by 1000) of episodes to run for. Ex: 12000
    exploration_schedule: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    gamma: float
        Discount Factor
    """
    ###############
    # RUN ENV     #
    ###############
    # Keep track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    n_thousand_episode = int(np.floor(num_episodes / 1000))
    # visits = np.zeros((n_thousand_episode, env.nS))
    total_timestep = 0
    meta_timestep = 0
    ctrl_timestep = defaultdict(int)

    # for i_thousand_episode in range(n_thousand_episode):
    # for i_episode in range(1000):
    episode_length = 0
    # visits[i_thousand_episode][current_state-1] += 1
    # encoded_current_state = one_hot_state(current_state)

    for i_episode in range(20000):
        state = env.reset()
        agent.first_goal_flag = 1
        Done = False
        last_goal = 0
        episode_reward = 0
        meta_timestep += 1
        meta_epsilon = exploration_schedule.value(total_timestep)
        goal = agent.select_goal(state, meta_epsilon)[0]  # one_hot:[0,1]
        while True:
            total_extrinsic_reward = 0.0
            state_0 = state
            while True:
                print("goal: ", goal)
                last_goal = goal
                total_timestep += 1
                episode_length += 1
                action_epsilon = exploration_schedule.value(total_timestep)
                action = agent.select_action(policy, env, state, goal, action_epsilon)
                env.goal = goal
                ### Step the env and store the transition
                next_state, extrinsic_reward, done, _ = env.step(action)
                episode_reward += extrinsic_reward
                intrinsic_reward = agent.get_intrinsic_reward(goal, next_state)

                agent.update_meta_controller(gamma)
                state = next_state

                total_extrinsic_reward += extrinsic_reward
                if done:
                    Done = True
                    break
                if goal == 0 and env.goal_0_reach == 1:
                    if agent.first_goal_flag == 1:
                        agent.first_goal_flag = 0
                    break
                if goal == 1 and env.goal_1_reach == 1:
                    break
                if episode_length%100 == 0:
                    agent.save_model()
            agent.meta_replay_memory.push(state_0, goal, next_state, total_extrinsic_reward, done)
            env.writer.add_scalars("reward",
                                   {"reward": total_extrinsic_reward}, total_timestep)
            if Done:
                break
            meta_epsilon = exploration_schedule.value(total_timestep)
            goal = agent.select_goal(state, meta_epsilon)[0]  # one_hot:[0,1]
            if goal == last_goal:
                goal += 1
                # break
        # Goal Finished
        env.writer.add_scalars("episode_reward",
                                {"episode_reward": episode_reward}, meta_timestep)
        print("---------------")
        print("episode_reward:  ", episode_reward)
        print("---------------")
        print("push buff number:", )


    return agent, stats

def hdqn_eval(
    env,
    agent,
    policy,
    num_episodes,
    exploration_schedule,
    gamma=1.0,
    ):
    state = env.reset()
    for i in range(2000):
        goal = agent.select_goal(state, 1)[0]
        while True:
            while True:
                print("goal: ", goal)
                action = agent.select_action(policy, env, state, goal, 0)
                env.goal = goal
                ### Step the env and store the transition
                next_state, extrinsic_reward, done, _ = env.step(action)
                state = next_state

                if done:
                    Done = True
                    break
                if goal == 0 and env.goal_0_reach == 1:
                    if agent.first_goal_flag == 1:
                        agent.first_goal_flag = 0
                    break
                if goal == 1 and env.goal_1_reach == 1:
                    break

            if Done:
                break
            goal = agent.select_goal(state, 1)[0]  # one_hot:[0,1]

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer, PPO, A2C
class HPPO(PPO):
    def __init__(self, policy, env, agent, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.agent = agent

    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "HPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = super()._setup_learn(
            total_timesteps, None, callback, 0, 0, None, reset_num_timesteps, tb_log_name
        )
        callback.on_training_start(locals(), globals())

        total_timestep = 0
        meta_timestep = 0
        for i_episode in range(total_timesteps // 100):
            obs, _ = self.env.reset()
            self.agent.first_goal_flag = 1
            done = False
            last_goal = 0
            episode_reward = 0
            meta_timestep += 1
            goal = self.agent.select_goal(obs)
            episode_length = 0
            while not done and episode_length < self.env.envs[0].max_steps:
                total_extrinsic_reward = 0.0
                inner_reward = 0.0
                state_0 = obs
                inner_done = False
                inner_step = 0
                while not inner_done and inner_step < 100:  # 内循环限制
                    total_timestep += 1
                    episode_length += 1
                    inner_step += 1
                    action = self.agent.select_action(obs, goal)  # 扩散模型生成动作
                    next_obs, extrinsic_reward, inner_done, truncated, info = self.env.step(action)
                    episode_reward += extrinsic_reward
                    intrinsic_reward = self.agent.get_intrinsic_reward(goal, next_obs)
                    inner_reward += intrinsic_reward

                    # 仅收集 meta 轨迹（controller 不更新）
                    self.agent.meta_buffer.add(
                        state_0,
                        goal,
                        inner_reward,
                        self._last_episode_starts[0],
                        next_obs,
                        inner_done
                    )
                    obs = next_obs
                    total_extrinsic_reward += extrinsic_reward
                    if inner_done or inner_step >= 100:
                        break
                    if goal == 0 and self.env.envs[0].goal_0_reach == 1:
                        if self.agent.first_goal_flag == 1:
                            self.agent.first_goal_flag = 0
                        break
                    if goal == 1 and self.env.envs[0].goal_1_reach == 1:
                        break

                # 更新 meta-controller
                self.agent.update_meta_controller(self.gamma)
                if inner_done:
                    done = True
                    break
                goal = self.agent.select_goal(obs)
                if goal == last_goal:
                    goal = (goal + 1) % 2  # 避免重复目标
                last_goal = goal

            if i_episode % log_interval == 0:
                print(f"Episode {i_episode}: Reward = {episode_reward}, Length = {episode_length}")

        callback.on_training_end()
        return self