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

        self.ob_dim = self.params['ob_dim']
        self.ac_dim = self.params['ac_dim']
        self.device = self.params['device']
        self.lr = self.params['learning_rate']
        self.discrete = self.params['discrete']
        self.gamma = self.params['gamma']
        self.epsilon = self.params['epsilon']
        self.value_loss_coeff = self.params['value_loss_coeff']
        self.entropy_coeff = self.params['entropy_coeff']
        self.standardize_advantages = self.params['standardize_advantages']
        self.num_target_updates = self.params['num_target_updates']
        self.num_grad_steps_per_target_update = self.params['num_grad_steps_per_target_update']
        self.n_layers = self.params['policy_n_layers']
        self.size = self.params['policy_layer_size']

        self.actor = PPOPolicy(self.ob_dim, self.ac_dim, self.n_layers, self.size, self.device,
                               self.lr, self.epsilon, self.value_loss_coeff, self.entropy_coeff)
        self.critic = PPOCritic(self.ob_dim, self.ac_dim, self.n_layers, self.size, self.device,
                                self.lr, self.gamma, self.num_target_updates, self.num_grad_steps_per_target_update)

        self.replay_buffer = ReplayBuffer()

    def compute_advantage(self, obs, next_obs, rews, terminals):
        assert type(obs) == type(next_obs) == type(rews) == type(terminals) == torch.Tensor, \
            'obs, next_obs, rews, terminals must be of type Tensor...'

        ob, next_ob, rew, done = map(lambda x: x.to(self.device), [obs, next_obs, rews, terminals])
        value = self.critic.value_func(ob).squeeze()
        next_value = self.critic.value_func(next_ob).squeeze() * (1 - done)
        advs = rew + self.gamma * next_value - value
        advs = advs.cpu().detach()

        if self.standardize_advantages:
            advs = advs.numpy()
            advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
            advs = torch.from_numpy(advs)

        return advs

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
                 epsilon,
                 value_loss_coeff,
                 entropy_coeff,
                 training=True,
                 discrete=True,
                 nn_baseline=False):

        # initialize network architecture

        self.device = device
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.discrete = discrete
        self.training = training
        # self.nn_baseline = nn_baseline

        policy_params = {}
        policy_params['input_dim'] = ob_dim
        policy_params['output_dim'] = ac_dim
        for i in range(n_layers-1):
            policy_params['l' + str(i+1) + '_dim'] = size
        self.policy = MLP(policy_params)
        params = list(self.policy.parameters())

        # if self.nn_baseline:
        #     baseline_params = {}
        #     baseline_params['input_dim'] = ob_dim
        #     baseline_params['output_dim'] = 1
        #     for i in range(n_layers - 1):
        #         baseline_params['l' + str(i + 1) + '_dim'] = size
        #     self.baseline = MLP(baseline_params)
        #     params += list(self.baseline.parameters())

        if self.training:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def get_log_prob(self, output, action):
        action = action.to(self.device)
        if self.discrete:
            output_probs = nn.functional.log_softmax(output).exp()
            return torch.distributions.Categorical(output_probs).log_prob(action)
        else:
            return torch.distributions.Normal(output[0], output[1]).log_prob(action).sum(-1)

    def update(self, obs, acs, advs, qvals=None):
        # perform backprop
        assert type(obs) == torch.Tensor and type(advs) == torch.Tensor, 'obs and advs must be of type Tensor...'
        out = self.policy(obs.to(self.device))
        logprob = self.get_log_prob(out, acs)

        self.optimizer.zero_grad()
        # PG objective
        policy_loss = torch.sum((-logprob * advs.to(self.device)))
        # TODO: change to PPO objective
        policy_loss.backward()

        # if self.nn_baseline:
        #     baseline_out = self.baseline(obs.to(self.device))
        #     baseline_target = torch.Tensor((qvals - qvals.mean()) / (qvals.std() + 1e-8)).to(self.device)
        #     baseline_criterion = nn.MSELoss()
        #     baseline_loss = baseline_criterion(baseline_out, baseline_target)
        #     baseline_loss.backward()

        self.optimizer.step()

        return policy_loss

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

    def __init__(self, ob_dim, ac_dim, n_layers, size, device, learning_rate, gamma,
                 num_target_updates, num_grad_steps_per_target_updates):

        # initialize network architecture for V function estimator

        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_target_updates = num_target_updates
        self.num_grad_steps_per_target_updates = num_grad_steps_per_target_updates

        vf_params = {}
        vf_params['input_dim'] = ob_dim
        vf_params['output_dim'] = ac_dim
        for i in range(n_layers - 1):
            vf_params['l' + str(i + 1) + '_dim'] = size
        self.value_func = MLP(vf_params)
        params = list(self.value_func.parameters())

        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update(self, obs, next_obs, rews, terminals):
        # perform backprop
        assert type(obs) == type(next_obs) == type(rews) == type(terminals) == torch.Tensor, \
            'obs, next_obs, rews, terminals must be of type Tensor...'
        ob, next_ob, rew, done = map(lambda x: x.to(self.device), [obs, next_obs, rews, terminals])

        for update in range(self.num_target_updates * self.num_grad_steps_per_target_updates):
            if update % self.num_grad_steps_per_target_updates == 0:
                next_value = self.value_func(next_ob).squeeze() * (1 - done)
                target_value = rew + self.gamma * next_value

            prediction = self.value_func(ob).squeeze()
            self.optimizer.zero_grad()
            loss = self.criterion(prediction, target_value)
            loss.backward()
            self.optimizer.step()
            target_value.detach_()

        return loss


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