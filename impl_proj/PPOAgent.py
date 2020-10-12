import torch
import gym
from sprites_env import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque
import numpy as np
from training_utils import init, AddBias
import time
import argparse
import wandb
import os
from models import Encoder, EncoderCNN, EncoderOracle, VAEReconstructionModel, VAERewardPredictionModel
from multi_threading_env import SubprocVecEnv, VecNormalize
from datetime import datetime


class PPOAgent:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            samples = rollouts.sample_trajectories(advantages, self.num_mini_batch)

            for sample in samples:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

                values, action_log_probs, dist_entropy = \
                    self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class FixedNormal(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, baseline, env_name, model_dir):
        super(Policy, self).__init__()
        base = MLPBase

        v = env_name.split('-')[1]
        baseline_map = {
            'cnn': {'class': EncoderCNN, 'model_path': None},
            'image-scratch': {'class': Encoder, 'model_path': None},
            'image-reconstruction': {'class': VAEReconstructionModel,
                                     'model_path': model_dir+'encoder_reconstruction_'+v+'.pt'},
            'reward-prediction': {'class': VAERewardPredictionModel,
                                  'model_path': model_dir+'encoder_reward_prediction_'+v+'.pt'},
            'oracle': {'class': EncoderOracle, 'model_path': None},
        }

        self.baseline = baseline
        self.encoder = baseline_map[baseline]['class']()

        if baseline_map[baseline]['model_path'] is not None:
            self.encoder.load_state_dict(torch.load(baseline_map[baseline]['model_path']))

        if baseline == 'cnn': # use flattened activation (hard-coded)
            self.base = base(784)
        else: # use embedding/obs directly
            self.base = base(obs_shape[0])

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        inputs = self.encoder(inputs, rl=True)
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action)
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        inputs = self.encoder(inputs, rl=True)
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        inputs = self.encoder(inputs, rl=True)
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__(hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, x):
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda, use_proper_time_limits=False):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * \
                    self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def sample_trajectories(self, advantages, num_mini_batch=None, mini_batch_size=None):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr',
        type=float,
        default=7e-4,
        help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=2000,
        help='number of steps of env steps collected in each iteration')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument(
        '--save-dir',
        default='./models/',
        help='directory to save agent logs (default: ./models/)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=5e6,
        help='number of environment steps to train (default: 20e6)')
    parser.add_argument(
        '--env-name',
        default='SpritesState-v0',
        help='environment to train on (choose from: Sprites-v0, SpritesState-v0, SpritesRepel-v0, SpritesRepelState-v0)')
    parser.add_argument(
        '--baseline',
        default='oracle',
        help='choose from: cnn, image-scratch, image-reconstruction, reward-prediction, oracle')
    parser.add_argument(
        '--load_model',
        action='store_true',
        help='load pre-trained actor-critic model')
    args = parser.parse_args()
    return args


def main_multi_thread(DEBUG=False):
    args = get_args()

    if not DEBUG:
        wandb.init(project="impl_jiefan")
        wandb.config.update(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def make_env():
        return gym.make(args.env_name)

    envs = [make_env for _ in range(args.num_processes)]
    envs = SubprocVecEnv(envs)
    envs = VecNormalize(envs, gamma=args.gamma)

    model_path = os.path.join(args.save_dir, args.env_name + '_' + args.baseline + '.pt')

    actor_critic = Policy(envs.observation_space.shape,
                          envs.action_space,
                          args.baseline,
                          args.env_name,
                          args.save_dir)

    if args.load_model and os.path.exists(model_path):
        print('Loaded model from %s\n' % model_path)
        actor_critic.load_state_dict(torch.load(model_path))

    actor_critic.to(device)

    agent = PPOAgent(actor_critic,
                     args.clip_param,
                     args.ppo_epoch,
                     args.num_mini_batch,
                     args.value_loss_coef,
                     args.entropy_coef,
                     lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(torch.from_numpy(obs))
    rollouts.to(device)

    episode_reward = torch.zeros(args.num_processes, 1)
    episode_rewards = deque(maxlen=50)

    print('Training starts at %s\n' % (datetime.now()))

    start = time.time()
    num_updates = int(args.num_env_steps // args.num_steps // args.num_processes)
    for itr in range(num_updates):
        for step in range(args.num_steps):
            # collect trajectories by running actor_critic
            with torch.no_grad():
                values, actions, action_log_probs = actor_critic.act(rollouts.obs[step])

            obs, rewards, dones, _ = envs.step(actions.squeeze(1).cpu().numpy())

            obs = torch.from_numpy(obs).float()
            rewards = torch.from_numpy(np.expand_dims(rewards, 1)).float()
            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
            rollouts.insert(obs, actions, action_log_probs, values, rewards, masks)

            # Notes: reset environment when done is handled in worker in multi_threading_env.py

            episode_reward += rewards
            finished_episode_rewards = ((1 - masks) * episode_reward).squeeze()
            finished_episode_rewards = finished_episode_rewards[finished_episode_rewards > 0]
            episode_rewards.extend(finished_episode_rewards)
            episode_reward *= masks

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        total_num_steps = (itr + 1) * args.num_processes * args.num_steps

        if (itr % args.save_interval == 0 or itr == num_updates - 1) and args.save_dir != "" and not DEBUG:
            torch.save(actor_critic.state_dict(), model_path)
            print('Actor critic saved to disk at itr %d , num timesteps %d , at %s\n'
                  % (itr, total_num_steps, datetime.now()))

        if itr % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()

            mean_reward = np.mean(episode_rewards)
            median_reward = np.median(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)

            print('Iteration %d, num timesteps %d, FPS %d, at %s\n'
                  'Last %d training episodes: mean reward %.4f, median reward %.4f, '
                  'min reward %.4f, max reward %.4f, value_loss %.4f, action_loss %.4f, dist_entropy %.4f\n' %
                  (itr, total_num_steps, int(total_num_steps / (end - start)), datetime.now(),
                   len(episode_rewards), mean_reward,  median_reward, min_reward,
                   max_reward, value_loss, action_loss, dist_entropy))

            if not DEBUG:
                wandb.log({'Total timesteps': total_num_steps,
                           'Mean reward': mean_reward,
                           'Median reward': median_reward,
                           'Min reward': min_reward,
                           'Max reward': max_reward})

        if args.eval_interval is not None and itr % args.eval_interval == 0 and len(episode_rewards) > 1:
            pass

    print('Training stops at %s\n' % (datetime.now()))


def main_single_thread(DEBUG=False):
    args = get_args()

    if not DEBUG:
        wandb.init(project="impl_jiefan")
        wandb.config.update(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(args.env_name)

    model_path = os.path.join(args.save_dir, args.env_name + '_' + args.baseline + '.pt')

    actor_critic = Policy(env.observation_space.shape,
                          env.action_space,
                          args.baseline,
                          args.env_name,
                          args.save_dir)

    if args.load_model and os.path.exists(model_path):
        print('Loaded model from %s\n' % model_path)
        actor_critic.load_state_dict(torch.load(model_path))

    actor_critic.to(device)

    agent = PPOAgent(actor_critic,
                     args.clip_param,
                     args.ppo_epoch,
                     args.num_mini_batch,
                     args.value_loss_coef,
                     args.entropy_coef,
                     lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, 1,
                              env.observation_space.shape, env.action_space)

    ob = env.reset()
    rollouts.obs[0].copy_(torch.from_numpy(ob))
    rollouts.to(device)

    episode_reward = torch.zeros(1)
    episode_rewards = deque(maxlen=200)

    print('Training starts at %s\n' % (datetime.now()))

    start = time.time()
    num_updates = int(args.num_env_steps // args.num_steps)

    for itr in range(num_updates):
        for step in range(args.num_steps):
            # collect trajectories by running actor_critic
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(rollouts.obs[step])

            ob, reward, done, _ = env.step(action.squeeze(1).cpu().numpy())

            if done:
                ob = env.reset()

            ob = torch.from_numpy(ob).float()
            reward = torch.FloatTensor([reward])
            mask = torch.FloatTensor([0.0] if done else [1.0])
            rollouts.insert(ob, action, action_log_prob, value, reward, mask)

            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward.item())
                episode_reward *= mask

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        total_num_steps = (itr + 1) * args.num_steps

        if (itr % args.save_interval == 0 or itr == num_updates - 1) and args.save_dir != "" and not DEBUG:
            torch.save(actor_critic.state_dict(), model_path)
            print('Actor critic saved to disk at itr %d , num timesteps %d , at %s\n'
                  % (itr, total_num_steps, datetime.now()))

        if itr % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()

            mean_reward = np.mean(episode_rewards)
            median_reward = np.median(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)

            print('Iteration %d, num timesteps %d, FPS %d, at %s\n'
                  'Last %d training episodes: mean reward %.4f, median reward %.4f, '
                  'min reward %.4f, max reward %.4f, value_loss %.4f, action_loss %.4f, dist_entropy %.4f\n' %
                  (itr, total_num_steps, int(total_num_steps / (end - start)), datetime.now(),
                   len(episode_rewards), mean_reward, median_reward, min_reward,
                   max_reward, value_loss, action_loss, dist_entropy))

            if not DEBUG:
                wandb.log({'Total timesteps': total_num_steps,
                           'Mean reward': mean_reward,
                           'Median reward': median_reward,
                           'Min reward': min_reward,
                           'Max reward': max_reward})

        if args.eval_interval is not None and itr % args.eval_interval == 0 and len(episode_rewards) > 1:
            pass

    print('Training stops at %s\n' % (datetime.now()))


if __name__ == "__main__":

    DEBUG = False
    main_multi_thread(DEBUG=DEBUG)
