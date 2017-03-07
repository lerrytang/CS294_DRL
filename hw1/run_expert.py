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
    parser.add_argument("--max_iter", type=int, default=100000,
			help="Number of training iterations")
    parser.add_argument("--init_lr", type=float, default=0.002,
			help="Initial learning rate")
    parser.add_argument("--reg_coef", type=float, default=0.0,
			help="Coefficient for L2 regularization")
    parser.add_argument("--logdir", type=str, default="log")
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

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
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
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

	### added for hw1
	if args.clone_expert:
            logdir = args.logdir
            logdir += "/" + args.envname + "_" + str(args.max_iter) + "_" + str(args.init_lr) + "_" + str(args.reg_coef)
            hw1_sol = HW1_sol(logdir, args.max_iter, args.init_lr, args.reg_coef)
            hw1_sol.clone_behavior(args.expert_policy_file,
                                   expert_data["observations"],
                                   expert_data["actions"])
            hw1_sol.test(env, args.num_rollouts, max_steps, args.render)

            print('[expert] mean return', np.mean(returns))
            print('[expert] std of return', np.std(returns))
	###


if __name__ == '__main__':
    main()
