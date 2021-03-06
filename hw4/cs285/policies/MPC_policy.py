import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        random_action_sequences = np.random.uniform(self.low, self.high, size=[num_sequences, horizon, self.ac_dim])
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []

        for model in self.dyn_models:
            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble
            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward (See files in envs to see how to call this)
            cur_obs = np.tile(obs, (self.N, 1))
            cur_rwds = np.zeros(self.N)
            for t in range(self.horizon):
                r_total, done = self.env.get_reward(cur_obs, candidate_action_sequences[:,t,:])
                cur_rwds += r_total
                cur_obs = model.get_prediction(cur_obs, candidate_action_sequences[:,t,:], self.data_statistics)
            predicted_rewards_per_ens.append(cur_rwds)

        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens,N] --> N
        predicted_rewards = np.mean(predicted_rewards_per_ens, axis=0)

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards)
        best_action_sequence = candidate_action_sequences[best_index]
        action_to_take = best_action_sequence[0]
        return action_to_take[None] # the None is for matching expected dimensions
