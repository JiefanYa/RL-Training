import tensorflow as tf

from .base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure.tf_utils import build_mlp


class FFModel(BaseModel):

    def __init__(self, sess, ac_dim, ob_dim, n_layers, size, learning_rate=0.001, scope='dyn_model'):
        super(FFModel, self).__init__()

        # init vars
        # self.env = env
        self.sess = sess
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.scope = scope

        # build TF graph
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.build_graph()
        self.define_train_op()

    #############################

    def build_graph(self):
        self.define_placeholders()
        self.define_forward_pass()

    def define_placeholders(self):

        self.obs_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        self.acs_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        self.delta_labels = tf.placeholder(shape=[None, self.ob_dim], name="labels", dtype=tf.float32)

        self.obs_mean_pl = tf.placeholder(shape=[self.ob_dim], name="obs_mean", dtype=tf.float32)
        self.obs_std_pl = tf.placeholder(shape=[self.ob_dim], name="obs_std", dtype=tf.float32)
        self.acs_mean_pl = tf.placeholder(shape=[self.ac_dim], name="acs_mean", dtype=tf.float32)
        self.acs_std_pl = tf.placeholder(shape=[self.ac_dim], name="acs_std", dtype=tf.float32)
        self.delta_mean_pl = tf.placeholder(shape=[self.ob_dim], name="delta_mean", dtype=tf.float32)
        self.delta_std_pl = tf.placeholder(shape=[self.ob_dim], name="delta_std", dtype=tf.float32)

    def define_forward_pass(self):
        # normalize input data to mean 0, std 1
        obs_unnormalized = self.obs_pl
        acs_unnormalized = self.acs_pl
        # Hint: Consider using the normalize function defined in infrastructure.utils for the following two lines
        obs_normalized = normalize(obs_unnormalized, self.obs_mean_pl, self.obs_std_pl)
        acs_normalized = normalize(acs_unnormalized, self.acs_mean_pl, self.acs_std_pl)

        # predicted change in obs
        concatenated_input = tf.concat([obs_normalized, acs_normalized], axis=1)
        # Hint: Note that the prefix delta is used in the variable below to denote changes in state, i.e. (s'-s)
        self.delta_pred_normalized = build_mlp(concatenated_input, self.ob_dim, self.scope, self.n_layers, self.size)
        self.delta_pred_unnormalized = unnormalize(self.delta_pred_normalized, self.delta_mean_pl, self.delta_std_pl)
        self.next_obs_pred = self.obs_pl + self.delta_pred_unnormalized

    def define_train_op(self):

        # normalize the labels
        self.delta_labels_normalized =  normalize(self.delta_labels, self.delta_mean_pl, self.delta_std_pl)

        # compared predicted deltas to labels (both should be normalized)
        self.loss = tf.losses.mean_squared_error(self.delta_labels_normalized, self.delta_pred_normalized)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    #############################

    def get_prediction(self, obs, acs, data_statistics):
        if len(obs.shape)>1:
            observations = obs
            actions = acs
        else:
            observations = obs[None]
            actions = acs [None]
        return self.sess.run([self.next_obs_pred], feed_dict={
            self.obs_pl: observations,
            self.acs_pl: actions,
            self.obs_mean_pl: data_statistics['obs_mean'],
            self.obs_std_pl: data_statistics['obs_std'],
            self.acs_mean_pl: data_statistics['acs_mean'],
            self.acs_std_pl: data_statistics['acs_std'],
            self.delta_mean_pl: data_statistics['delta_mean'],
            self.delta_std_pl: data_statistics['delta_std']
        })[0]

    def update(self, observations, actions, next_observations, data_statistics):
        # train the model
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                                    self.obs_pl: observations,
                                    self.acs_pl: actions,
                                    self.delta_labels: next_observations - observations,
                                    self.obs_mean_pl: data_statistics['obs_mean'],
                                    self.obs_std_pl: data_statistics['obs_std'],
                                    self.acs_mean_pl: data_statistics['acs_mean'],
                                    self.acs_std_pl: data_statistics['acs_std'],
                                    self.delta_mean_pl: data_statistics['delta_mean'],
                                    self.delta_std_pl: data_statistics['delta_std']
                                })
        return loss