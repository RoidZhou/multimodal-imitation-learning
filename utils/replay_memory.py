import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.observation = {}
        self.first_frame_observation = 1
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def prepare_observation(self, obs_init, horizon, batch_size, n_obs_steps):
        """
        input: []
        output: []

        Returns:

        """
        obs = obs_init[0]
        if self.first_frame_observation == 1:
            images = np.zeros((batch_size, horizon, *np.transpose(obs['image'], (2, 0, 1)).shape))
            images_hand = np.zeros((batch_size, horizon, *np.transpose(obs['image_hand'], (2, 0, 1)).shape))
            depths = np.zeros((batch_size, horizon, *np.transpose(obs['depth'], (2, 0, 1)).shape))
            agent_pos = np.zeros((batch_size, horizon, *obs['agent_pos'].shape))
            force = np.zeros((batch_size, horizon, *obs['force'].shape))
            for j in range(batch_size):
                for i in range(horizon):
                    images[j, i, ...] = np.transpose(obs_init[j]['image'], (2, 0, 1))
                    images_hand[j, i, ...] = np.transpose(obs_init[j]['image_hand'], (2, 0, 1))
                    depths[j, i, ...] = np.transpose(obs_init[j]['depth'], (2, 0, 1))
                    agent_pos[j, i, ...] = obs_init[j]['agent_pos']
                    force[j, i, ...] = obs_init[j]['force']

            self.observation['images'] = images
            self.observation['images_hand'] = images_hand
            self.observation['depths'] = depths
            self.observation['agent_pos'] = agent_pos
            self.observation['force'] = force
            self.first_frame_observation = 0
        else:
            for j in range(batch_size):
                self.observation['images'][j, 0:n_obs_steps - 1, ...] = self.observation['images'][j, 1:n_obs_steps, ...]
                self.observation['images_hand'][j, 0:n_obs_steps - 1, ...] = self.observation['images_hand'][j, 1:n_obs_steps, ...]
                # observation['depths'][j, 0:n_obs_steps - 1, ...] = observation['depths'][0, 1:n_obs_steps, ...]
                self.observation['agent_pos'][j, 0:n_obs_steps - 1, ...] = self.observation['agent_pos'][j, 1:n_obs_steps, ...]

                self.observation['images'][j, n_obs_steps - 1, ...] = np.transpose(obs_init[j]['image'], (2,0,1))
                self.observation['images_hand'][j, n_obs_steps - 1, ...] = np.transpose(obs_init[j]['image_hand'], (2,0,1))
                # observation['depths'][j, n_obs_steps - 1, ...] = np.transpose(obs_init[j]['depth'], (2,0,1))
                self.observation['agent_pos'][j, n_obs_steps - 1, ...] = obs_init[j]['agent_pos']

        return self.observation

    def sample(self, func, horizon, batch_size, n_obs_steps):
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = zip(*random.sample(self.memory, batch_size))
        curr_state_batch = self.prepare_observation(state_batch, horizon, batch_size, n_obs_steps)
        state_batch = func(curr_state_batch)
        goal_batch =  np.array([t.numpy() for t in goal_batch])

        next_state_batch = self.prepare_observation(next_state_batch, horizon, batch_size, n_obs_steps)
        next_state_batch = func(next_state_batch)

        ex_reward_batch =  np.array(ex_reward_batch)

        done_mask = np.array(done_mask)
        return state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask

    def __len__(self):
        return len(self.memory)
