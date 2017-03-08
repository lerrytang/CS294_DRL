#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from hw1_sol import HW1_sol


def expert_test(env, num_rollouts, policy_fn, max_steps, render=False):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('[expert] mean return: {0:.2f}'.format(np.mean(returns)))
    print('[expert] std of return: {0:.2f}'.format(np.std(returns)))

    return returns, observations, actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    ### added for hw1
    parser.add_argument("--clone_expert", action="store_true")
    parser.add_argument("--train_epoch", type=int, default=10,
			help="Number of epoches to train")
    parser.add_argument("--init_lr", type=float, default=0.002,
			help="Initial learning rate")
    parser.add_argument("--reg_coef", type=float, default=0.0,
			help="Coefficient for L2 regularization")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument("--dagger_iter", type=int, default=20,
			help="Number of dagger iterations")
    ###
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        returns, observations, actions = expert_test(env,
                                                     args.num_rollouts,
                                                     policy_fn,
                                                     max_steps,
                                                     args.render)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        print('#samples={}'.format(len(actions)))

	if args.clone_expert:
            logdir = args.logdir
            logdir += "/" + args.envname + "_" + str(args.num_rollouts) + "_" +\
                        str(args.train_epoch) + "_" + str(args.init_lr) + "_" +\
                        str(args.reg_coef)
            bc = HW1_sol(args.expert_policy_file,logdir, args.train_epoch, args.init_lr, args.reg_coef)
            bc.clone_behavior(expert_data["observations"],
                              expert_data["actions"])
            bc.test(env, args.num_rollouts, max_steps, args.render)
            expert_test(env, args.num_rollouts, policy_fn, max_steps, args.render)

        if args.dagger:
            logdir = args.logdir
            logdir += "/DAgger_" + args.envname + "_" + str(args.num_rollouts) + "_" +\
                        str(args.train_epoch) + "_" + str(args.init_lr) + "_" +\
                        str(args.reg_coef)

            bc = HW1_sol(args.expert_policy_file, logdir, args.train_epoch, args.init_lr, args.reg_coef, "BC")
            dagger = HW1_sol(args.expert_policy_file, logdir, args.train_epoch, args.init_lr, args.reg_coef, "DAgger")

            writer = tf.summary.FileWriter(logdir, tf_util.get_session().graph)
            bc.writer = writer
            dagger.writer = writer

            bc_obs = expert_data["observations"]
            bc_act = expert_data["actions"]
            n_total = bc_obs.shape[0]
            n_data_each_iter = int(np.round(1.0 * n_total / args.dagger_iter))
            dagger_obs = bc_obs[:n_data_each_iter]
            dagger_act = bc_act[:n_data_each_iter]

            n_dagger_iter = 0
            while n_dagger_iter < args.dagger_iter:

                # train bc
                bc.clone_behavior(bc_obs, bc_act)
                # train dagger
                if n_dagger_iter == 0:
                    dagger.clone_behavior(dagger_obs, dagger_act)
                else:
                    dagger.train(dagger_obs, dagger_act)

                print("Test at dagger_iter={}".format(n_dagger_iter))

                # test expert
                expert_test(env, args.num_rollouts, policy_fn, max_steps, args.render)
                # test bc
                bc.test(env, args.num_rollouts, max_steps, args.render)
                # test dagger
                bc_obs = dagger.test(env, args.num_rollouts, max_steps, args.render)
                bc_act = []
                for obs in bc_obs:
                    bc_act.append(policy_fn(obs[None,:]))

                bc_obs = np.array(bc_obs)
                bc_act = np.array(bc_act)
                dagger_obs = np.concatenate([dagger_obs, bc_obs[:n_data_each_iter]])
                dagger_act = np.concatenate([dagger_act, bc_act[:n_data_each_iter]])

                n_dagger_iter += 1


if __name__ == '__main__':
    main()
