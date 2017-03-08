import tensorflow as tf
import numpy as np
import tf_util
import pickle


class HW1_sol:

    def __init__(self, filename, logdir, train_epoch, init_lr, reg_coef, name="BC"):
        self._logdir = logdir
        self._train_epoch = train_epoch
        self._init_lr = init_lr
        self._reg_coef = reg_coef
        self._name = name

        self.obs = None
        self.est_act = None
        self.exp_act = None
        self.writer = None

        self.summary_tensors = []

        with tf.variable_scope(self._name):
            # copy network structure
            self._build_network(filename)
            # add L2 loss
            self._add_loss()

        net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        with tf.variable_scope(self._name + "_Adam"):
            # add optimizer
            opt = tf.train.AdamOptimizer(learning_rate=self._init_lr, name=self._name + "_optimizer")
            self.train_op = opt.minimize(self.loss, var_list=net_vars)

        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name + "_Adam")
        all_vars = net_vars + opt_vars
        with tf.variable_scope(self._name + "_init"):
            self.init_op = tf.variables_initializer(all_vars)
            # add summary
        self.merged = tf.summary.merge(self.summary_tensors)

    def clone_behavior(self, obs_data, act_data):
        # initialize all variables
        tf_util.get_session().run(self.init_op)
        # training
        self.train(obs_data, act_data)
    
    def _build_network(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())
    
        # assert len(data.keys()) == 2
        nonlin_type = data['nonlin_type']
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
        policy_params = data[policy_type]
    
        # Keep track of input and output dims
        def build_policy(obs_bo):
            def read_layer(l, x):
                assert list(l.keys()) == ['AffineLayer']
                assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
                return tf.layers.dense(inputs=x,
                                       units=l["AffineLayer"]["W"].shape[1])
                
            def apply_nonlin(x):
                if nonlin_type == 'lrelu':
                    return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
                elif nonlin_type == 'tanh':
                    return tf.tanh(x)
                else:
                    raise NotImplementedError(nonlin_type)
    
            # Build the policy. First, observation normalization.
            assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
            obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
            obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
            obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
            print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
            normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6)
    
            curr_activations_bd = normedobs_bo
    
            # Hidden layers next
            assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
            layer_params = policy_params['hidden']['FeedforwardNet']
            for layer_name in sorted(layer_params.keys()):
                l = layer_params[layer_name]
                curr_activations_bd = apply_nonlin(read_layer(l, curr_activations_bd))
    
            # Output layer
            output_bo = read_layer(policy_params['out'], curr_activations_bd)
            return output_bo
    
        self.obs = tf.placeholder(tf.float32, [None, None], name="obs")
        self.est_act = build_policy(self.obs)
    
    def _add_loss(self):
        self.exp_act = tf.placeholder(tf.float32, self.est_act.get_shape().as_list(), name="exp_act")
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square([self.est_act - self.exp_act]), axis=1))
        self.summary_tensors.append(tf.summary.scalar("loss/Regress loss", self.loss))
    
    def train(self, obs_data, act_data, batch_size=64):
    
        print("obs_data.shape:", obs_data.shape)
        print("act_data.shape:", act_data.shape)
        n_total = obs_data.shape[0]
        assert n_total == act_data.shape[0], "Sizes do not match ({}vs{})".format(n_total, act_data.shape[0])
        print("training data size = {}".format(n_total))
        iter_per_epoch = int(np.ceil(1.0 * self._train_epoch * n_total / batch_size))
        print("iter_per_epoch = {}".format(iter_per_epoch))
        n_epoch = 0
        while n_epoch < self._train_epoch:
            n_iter = 0
            while n_iter < iter_per_epoch:
                rand_idx = np.random.choice(n_total, size=batch_size, replace=False if batch_size<n_total else True)
                obs_batch = obs_data[rand_idx]
                act_batch = act_data[rand_idx]
                train_loss, summary, _ = tf_util.get_session().run([self.loss,
                                                                    self.merged,
                                                                    self.train_op],
                        feed_dict={self.obs: obs_batch, self.exp_act: act_batch.squeeze()})

                n_iter += 1
            n_epoch += 1
            print("epoch={0}/{1} loss={2:.4f}".format(n_epoch, self._train_epoch, train_loss))
            if self.writer is not None:
                self.writer.add_summary(summary, global_step=n_epoch)
    
    def test(self, env, num_rollouts, max_steps, render=False):
        returns = []
        observations = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = tf_util.get_session().run(self.est_act,
                                            feed_dict={self.obs: (obs[None,:])})
                observations.append(obs)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)
    
        print('[{0}] mean return: {1:.2f}'.format(self._name, np.mean(returns)))
        print('[{0}] std of return: {1:.2f}'.format(self._name, np.std(returns)))

        return observations


