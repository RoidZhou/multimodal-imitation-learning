import numpy as np
import random
from collections import namedtuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory, Transition
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.vision.multimodal_obs_encoder import MultiModalObsEncoder, Encoder2rlLayer
from diffusion_policy_3d.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
import yaml
from diffusion_policy_3d.model.vision.model_getter import get_resnet
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3 import PPO
from gym import error, spaces, utils
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

class MetaController(nn.Module):
    def __init__(self, in_features=6, out_features=6):
        """
        Initialize a Meta-Controller of Hierarchical DQN network for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Controller(nn.Module):
    def __init__(self, in_features=12, out_features=2):
        """
        Initialize a Controller(given goal) of h-DQN for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class hrl_agent():
    """
    The Hierarchical-DQN Agent
    Parameters
    ----------
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        num_goal: int
            The number of goal that agent can choose from
        num_action: int
            The number of action that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
    """
    def __init__(self,
                 meta_policy,
                 optimizer_spec,
                 obs_encoder: MultiModalObsEncoder,
                 pretrained_mode_path,
                 num_goal=6,
                 num_action=2,
                 replay_memory_size=10000,
                 batch_size=32,
                 device='cuda:0'):
        ###############
        # BUILD MODEL #
        ###############
        self.normalizer = LinearNormalizer()
        # obs_feature_dim = obs_encoder.output_shape()[0]
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        # Construct meta-controller and controller
        self.device = torch.device(device)
        self.obs_encoder = obs_encoder.to(self.device)
        model_dict = torch.load(pretrained_mode_path)
        features_dict = {k.replace('obs_encoder.', ''): v for k, v in model_dict.items()
                         if k.startswith('obs_encoder.')}
        self.meta_policy = meta_policy
        self.obs_encoder.load_state_dict(features_dict)
        self.target_obs_encoder = obs_encoder.to(self.device)
        self.target_obs_encoder.load_state_dict(features_dict)
        fc = Encoder2rlLayer(1027, 128, 3)  # 定义全连接层
        target_fc = Encoder2rlLayer(1027, 128, 3)  # 定义全连接层
        self.fc = fc.to(self.device)
        self.target_fc = target_fc.to(self.device)
        # self.meta_controller = MetaController().type(dtype)
        # self.target_meta_controller = MetaController().type(dtype)
        self.controller = Controller().type(dtype)
        self.target_controller = Controller().type(dtype)
        # Construct the optimizers for meta-controller and controller
        # self.meta_optimizer = optimizer_spec.constructor(self.meta_controller.parameters(), **optimizer_spec.kwargs)
        # self.ctrl_optimizer = optimizer_spec.constructor(self.controller.parameters(), **optimizer_spec.kwargs)
        self.obs_encoder_optimizer = optimizer_spec.constructor(self.obs_encoder.parameters(), **optimizer_spec.kwargs)
        self.fc_optimizer = optimizer_spec.constructor(self.fc.parameters(), **optimizer_spec.kwargs)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayMemory(replay_memory_size)
        self.ctrl_replay_memory = ReplayMemory(replay_memory_size)
        self.observation = {}
        self.actions = np.zeros(7)
        self.horizon = 3
        self.n_obs_steps = 2
        self.first_frame_observation = 1
        self.first_goal_flag = 1
        self.update_count = 0
        self.target_update_frequency = 200

    def get_intrinsic_reward(self, goal, state):
        return state

    def select_goal(self, obs, epilson):
        sample = random.random()
        if self.first_goal_flag == 1:
            return torch.IntTensor([0])
        else:
            if sample > epilson:
                # state = torch.from_numpy(state).type(dtype)
                images = np.zeros((1, self.horizon, *np.transpose(obs['image'], (2, 0, 1)).shape))
                images_hand = np.zeros((1, self.horizon, *np.transpose(obs['image_hand'], (2, 0, 1)).shape))
                depths = np.zeros((1, self.horizon, *np.transpose(obs['depth'], (2, 0, 1)).shape))
                agent_pos = np.zeros((1, self.horizon, *obs['agent_pos'].shape))
                force = np.zeros((1, self.horizon, *obs['force'].shape))
                for i in range(self.horizon):
                    images[0, i, ...] = np.transpose(obs['image'], (2, 0, 1))
                    images_hand[0, i, ...] = np.transpose(obs['image_hand'], (2, 0, 1))
                    depths[0, i, ...] = np.transpose(obs['depth'], (2, 0, 1))
                    agent_pos[0, i, ...] = obs['agent_pos']
                    force[0, i, ...] = obs['force']
                observation = {}
                observation['images'] = images
                observation['images_hand'] = images_hand
                observation['depths'] = depths
                observation['agent_pos'] = agent_pos
                observation['force'] = force

                # 转换为编码器需要的格式
                encoder_input = self.prepare_eval_observation(observation)
                encoder_batch = dict_apply(encoder_input, lambda x: x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                with torch.no_grad():  # 确保不计算梯度（推理模式）
                    nobs_features = self.obs_encoder(encoder_batch)
                    nobs_features = self.meta_policy.predict(nobs_features).data.max(1)[1]
                    return nobs_features.cpu()  # 直接使用 tensor
            else:
                return torch.IntTensor([random.randrange(self.num_action)])

    def select_action(self, policy, env_pose, env_orien, obs, action):
        sample = random.random()

        # 获取值为 1 的索引（即预测类别）
        # class_indices = torch.argmax(goal, dim=-1)  # shape: (B,)
        class_indices = action  # shape: (B,)
        # if sample < action_epsilon:
        #     class_indices = 1-class_indices
        if class_indices == 0 or class_indices == 1:
            # self.prepare_select_action_observation()

            if self.first_frame_observation == 1:
                images = np.zeros((1, self.horizon, *np.transpose(obs['image'], (2, 0, 1)).shape))
                images_hand = np.zeros((1, self.horizon, *np.transpose(obs['image_hand'], (2, 0, 1)).shape))
                depths = np.zeros((1, self.horizon, *np.transpose(obs['depth'], (2, 0, 1)).shape))
                agent_pos = np.zeros((1, self.horizon, *obs['agent_pos'].shape))
                force = np.zeros((1, self.horizon, *obs['force'].shape))

                for i in range(self.horizon):
                    images[0, i, ...] = np.transpose(obs['image'], (2, 0, 1))
                    images_hand[0, i, ...] = np.transpose(obs['image_hand'], (2, 0, 1))
                    depths[0, i, ...] = np.transpose(obs['depth'], (2, 0, 1))
                    agent_pos[0, i, ...] = obs['agent_pos']
                    force[0, i, ...] = obs['force']

                self.observation['images'] = images
                self.observation['images_hand'] = images_hand
                self.observation['depths'] = depths
                self.observation['agent_pos'] = agent_pos[:,:,3:7]
                self.first_frame_observation = 0
            else:
                self.observation['images'][0, 0:self.n_obs_steps - 1, ...] = self.observation['images'][0, 1:self.n_obs_steps, ...]
                self.observation['images_hand'][0, 0:self.n_obs_steps - 1, ...] = self.observation['images_hand'][0, 1:self.n_obs_steps, ...]
                # observation['depths'][0, 0:n_obs_steps - 1, ...] = observation['depths'][0, 1:n_obs_steps, ...]
                self.observation['agent_pos'][0, 0:self.n_obs_steps - 1, ...] = self.observation['agent_pos'][0, 1:self.n_obs_steps, ...]

                self.observation['images'][0, self.n_obs_steps - 1, ...] = np.transpose(obs['image'], (2,0,1))
                self.observation['images_hand'][0, self.n_obs_steps - 1, ...] = np.transpose(obs['image_hand'], (2,0,1))
                # observation['depths'][0, n_obs_steps - 1, ...] = np.transpose(obs['depth'], (2,0,1))
                self.observation['agent_pos'][0, self.n_obs_steps - 1, ...] = obs['agent_pos'][3:7]

            self.actions[0:3] = env_pose
            isinstance(policy, DiffusionUnetImagePolicy)
            actions = policy.predict_action(self.observation)['action'].cpu().detach().numpy()
            self.actions[3:7] = actions[:,2,:] # 选择最新一帧
        else:
            self.actions[0:3] = env_pose
            self.actions[3:7] = env_orien
            self.actions[2] -= 0.001

        return self.actions

    # 准备输入数据
    def prepare_eval_observation(self, obs):
        # 将numpy数组转换为torch张量

        observation = {
            'images': torch.from_numpy(obs['images'][0, 0]).float().unsqueeze(0),
            'images_hand': torch.from_numpy(obs['images_hand'][0, 0]).float().unsqueeze(0),
            'depths': torch.from_numpy(obs['depths'][0, 0]).float().unsqueeze(0),  # 深度图添加通道维度
            'agent_pos': torch.from_numpy(obs['agent_pos'][0, 0]).float().unsqueeze(0),
            'force': torch.from_numpy(obs['force'][0, 0]).float().unsqueeze(0)
        }
        return observation

    # 准备输入数据
    def prepare_train_observation(self, obs):
        # 将numpy数组转换为torch张量

        observation = {
            'images': torch.from_numpy(obs['images'][:, 0]).float(),
            'images_hand': torch.from_numpy(obs['images_hand'][:, 0]).float(),
            'depths': torch.from_numpy(obs['depths'][:, 0]).float(),  # 深度图添加通道维度
            'agent_pos': torch.from_numpy(obs['agent_pos'][:, 0]).float(),
            'force': torch.from_numpy(obs['force'][:, 0]).float()
        }
        return observation

    def one_hot_state(self, state):
        vector = np.zeros(2)
        vector[state - 1] = 1.0
        return np.expand_dims(vector, axis=0)

    def prepare_select_action_observation(self, obs, horizon, batch_size, n_obs_steps):
        """
        input: []
        output: []

        Returns:

        """
        # obs = obs_init[0]
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

    def save_model(self):
        self.meta_policy.save("meta_policy.zip")

