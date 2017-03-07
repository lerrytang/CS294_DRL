import tensorflow as tf
import numpy as np
import tf_util
import pickle


class HW1_sol:

    def __init__(self, logdir, max_iter, init_lr, reg_coef):
        self._logdir = logdir
        self._max_iter = max_iter
        self._init_lr = init_lr
        self._reg_coef = reg_coef

        self.obs = None
        self.est_act = None
        self.exp_act = None

    def clone_behavior(self, filename, obs_data, act_data):
        # copy network structure
        self.obs, self.est_act = self._build_network(filename)
        # add L2 loss
        self.loss = self._add_loss()
        # add optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self._init_lr).minimize(self.loss)
        # initialize all variables
        tf_util.initialize()
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
                                       units=l["AffineLayer"]["W"].shape[1],
                                       kernel_regularizer=lambda x: tf_util.l2loss([x]))
                
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
    
        obs_bo = tf.placeholder(tf.float32, [None, None])
        a_ba = build_policy(obs_bo)
        return obs_bo, a_ba
    
    def _add_loss(self):
        self.exp_act = tf.placeholder(tf.float32, self.est_act.get_shape().as_list())
        loss = tf.reduce_mean(tf.reduce_sum(tf.square([self.est_act - self.exp_act]), axis=1))
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = loss + self._reg_coef * reg_loss
        tf.summary.scalar("loss/Regress loss", loss)
        tf.summary.scalar("loss/Regularization loss", reg_loss)
        tf.summary.scalar("loss/Total loss", total_loss)
        return total_loss
    
    def train(self, obs_data, act_data, batch_size=64):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self._logdir, tf_util.get_session().graph)
    
        n_total = obs_data.shape[0]
        assert n_total == act_data.shape[0], "Sizes do not match ({}vs{})".format(n_total, act_data.shape[0])
        print("training data size = {}".format(n_total))
        n_iter = 0
        while n_iter < self._max_iter:
            rand_idx = np.random.choice(n_total, size=batch_size, replace=False)
            obs_batch = obs_data[rand_idx]
            act_batch = act_data[rand_idx]
            train_loss, summary, _ = tf_util.get_session().run([self.loss,
                                                                merged,
                                                                self.train_op],
                    feed_dict={self.obs: obs_batch, self.exp_act: act_batch.squeeze()})
            n_iter += 1
            if n_iter % 500 == 0:
                print("{0}/{1} loss={2:.4f}".format(n_iter, self._max_iter, train_loss))
                writer.add_summary(summary, global_step=n_iter-1)
    
    def test(self, env, num_rollouts, max_steps, render=False):
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = tf_util.get_session().run(self.est_act,
                                            feed_dict={self.obs: (obs[None,:])})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
    
        print('[immitation] mean return', np.mean(returns))
        print('[immitation] std of return', np.std(returns))


