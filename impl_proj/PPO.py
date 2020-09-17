# import sprites_env
from training_utils import MLP
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from training_utils import *


class RL_Trainer(object):

    def __init__(self, params):

        self.params = params
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        # env
        self.env = gym.make(self.params['env_name']) # Sprites-v1
        self.env.seed(seed)

        # agent
        self.agent = PPOAgent(self.env, self.params)

    def train(self, n_itr, policy):
        # perform RL training loop
        self.start_time = time.time()

        for itr in range(n_itr):
            # collect trajectories
            # add to replay buffer
            # train agent
            # log
            return

    def collect_trajectories(self, *args):
        # collect trajectories
        return

    def train_agent(self):
        # update agent
        return


class PPOAgent(object):

    def __init__(self, env, params):

        self.params = params
        self.env = env

        args = None
        self.actor = PPOPolicy(args)
        self.critic = PPOCritic(args)

        self.replay_buffer = ReplayBuffer(args)

    def compute_advantage(self, *args):
        # compute A*
        return

    def train(self, *args):
        # call actor update
        return


class PPOPolicy(object):

    def __init__(self,
                 ob_dim,
                 ac_dim,
                 n_layers,
                 size,
                 device,
                 learning_rate,
                 discrete= False,
                 training=True,
                 nn_baseline=False):

        # initialize network architecture

        self.device = device
        self.learning_rate = learning_rate
        self.discrete = discrete
        self.training = training
        self.nn_baseline = nn_baseline

        policy_params = {}
        policy_params['input_dim'] = ob_dim
        policy_params['output_dim'] = ac_dim
        for i in range(n_layers-1):
            policy_params['l' + str(i+1) + '_dim'] = size
        self.policy = MLP(policy_params)
        params = list(self.policy.parameters())

        if self.nn_baseline:
            baseline_params = {}
            baseline_params['input_dim'] = ob_dim
            baseline_params['output_dim'] = 1
            for i in range(n_layers - 1):
                baseline_params['l' + str(i + 1) + '_dim'] = size
            self.baseline = MLP(baseline_params)
            params += list(self.baseline.parameters())

        if self.training:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def get_log_prob(self, output, action):
        action = action.to(self.device)
        if self.discrete:
            output_probs = nn.functional.log_softmax(output).exp()
            return torch.distributions.Categorical(output_probs).log_prob(action)
        else:
            return torch.distributions.Normal(output[0], output[1]).log_prob(action).sum(-1)

    def update(self, obs, acs, advs):
        # perform backprop
        assert type(obs) == torch.Tensor, 'obs must be of type Tensor...'
        out = self.policy(obs.to(self.device))
        logprobs = self.get_log_prob(out, acs)

        self.optimizer.zero_grad()
        # TODO


    def get_action(self, obs):
        # query for action
        assert type(obs) == torch.Tensor, 'obs must be of type Tensor...'
        out = self.policy(obs.to(self.device))
        if self.discrete:
            action_probs = nn.functional.log_softmax(out).exp()
            return torch.multinomial(action_probs,1).cpu().detach().numpy()[0]
        else:
            return torch.normal(out[0],out[1]).cpu().detach().numpy()


class PPOCritic(object):

    def __init__(self):
        # initialize network architecture for V function
        return

    def update(self, *args):
        # perform backprop
        return


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths):

        for path in paths:
            self.paths.append(path)

        obs, acs, next_obs, terminals, rews = convert_paths_to_components(paths)

        if self.obs is None:
            self.obs = obs[-self.max_size:]
            self.acs = acs[-self.max_size:]
            self.next_obs = next_obs[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.rews = rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, obs])[-self.max_size:]
            self.acs = np.concatenate([self.acs, acs])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_obs])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            self.rews = np.concatenate([self.rews, rews])[-self.max_size:]

    def sample_random_rollouts(self, num):
        indices = np.random.permutation(len(self.paths))[:num]
        return self.paths[indices]

    def sample_recent_rollouts(self, num=1):
        return self.paths[-num:]

    def sample_random_data(self, batch_size):
        assert self.obs.shape[0] == self.acs.shape[0] == self.next_obs.shape[0] \
               == self.terminals.shape[0] == self.rews.shape[0]

        indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[indices], self.acs[indices], self.next_obs[indices], \
               self.terminals[indices], self.rews[indices]

    def sample_recent_data(self, batch_size=1):
        return self.obs[-batch_size:], self.acs[-batch_size:], self.next_obs[-batch_size:], \
               self.terminals[-batch_size:], self.rews[-batch_size:]

    def __len__(self):
        return len(self.obs)