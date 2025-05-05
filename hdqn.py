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

def hdqn_learning(
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

    for i_thousand_episode in range(n_thousand_episode):
        for i_episode in range(1000):
            episode_length = 0
            obs = env.reset()
            obs0 = obs
            # visits[i_thousand_episode][current_state-1] += 1
            # encoded_current_state = one_hot_state(current_state)

            Done = False
            while True:
                obs = env.reset()
                obs0 = obs
                meta_timestep += 1
                # Get annealing exploration rate (epislon) from exploration_schedule
                # meta_epsilon = exploration_schedule.value(total_timestep)
                # goal = agent.select_goal(obs, meta_epsilon)[0] # one_hot:[0,1]
                # print("goal: ", goal)
                # goal = agent.one_hot_state(goal)

                total_extrinsic_reward = 0.0
                goal_reached = False
                while True:
                    meta_epsilon = exploration_schedule.value(total_timestep)
                    goal = agent.select_goal(obs, meta_epsilon)[0]  # one_hot:[0,1]
                    print("goal: ", goal)
                    total_timestep += 1
                    episode_length += 1
                    # ctrl_timestep[goal] += 1
                    # Get annealing exploration rate (epislon) from exploration_schedule
                    # ctrl_epsilon = exploration_schedule.value(total_timestep)
                    # joint_state_goal = np.concatenate([encoded_current_state, encoded_goal], axis=1)

                    action = agent.select_action(policy, env, obs, goal)
                    env.goal = goal
                    ### Step the env and store the transition
                    next_state, extrinsic_reward, done, _ = env.step(action)
                    obs = next_state
                    # Update statistics
                    # stats.episode_rewards[i_thousand_episode*1000 + i_episode] += extrinsic_reward
                    # stats.episode_lengths[i_thousand_episode*1000 + i_episode] = episode_length
                    # visits[i_thousand_episode][next_state-1] += 1

                    # encoded_next_state = one_hot_state(next_state)
                    intrinsic_reward = agent.get_intrinsic_reward(goal, next_state)
                    # goal_reached = next_state == goal
                    #
                    # joint_next_state_goal = np.concatenate([encoded_next_state, encoded_goal], axis=1)
                    # agent.ctrl_replay_memory.push(joint_state_goal, action, joint_next_state_goal, intrinsic_reward, done)
                    # # Update Both meta-controller and controller
                    agent.update_meta_controller(gamma)

                    # agent.obs_encoder
                    # agent.update_controller(gamma)
                    #
                    total_extrinsic_reward += extrinsic_reward
                    if done:
                        break
                    if episode_length%100 == 0:
                        agent.save_model()
                    # current_state = next_state
                    # encoded_current_state = encoded_next_state
                # Goal Finished
                print("---------------")
                print("extrinsic_reward:  ", total_extrinsic_reward)
                print("---------------")
                agent.meta_replay_memory.push(obs0, goal, obs, total_extrinsic_reward, done)
                print("push buff number:", )

        print("step : ", i_thousand_episode)

    return agent, stats
