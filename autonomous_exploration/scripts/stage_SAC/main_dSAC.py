#!/usr/bin/env python

import rospy
from ros_stage_env import StageEnvironment
from config import Config
import numpy as np
import matplotlib.pyplot as plt
import os
 
from Discrete_SAC_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 4
RUNS = 5
EPISODES_PER_RUN = 1000
STEPS_PER_EPISODE = 10
LOAD_MODELS = False

if __name__ == "__main__":
    args = Config().parse()

    env = StageEnvironment(args, flag_rnd=False)
    while(env.observation_space.shape[0] == 0):
        rospy.sleep(1)

    obs,_ = env.reset()


    obs_shape = [obs[0].shape,obs[1].shape]
    action_space = env.action_space

    flag_first = True
    changed_pose = []


    agent_results = []
    for run in range(RUNS):
        agent = SACAgent(obs_shape,action_space)

        if LOAD_MODELS:
            print("Loading Models")
            agent.load_net()

        run_results = []
        # for episode_number in range(MAX_EPISODES):
        episode_number = 0
        while episode_number < EPISODES_PER_RUN and not rospy.is_shutdown():

            if flag_first:
                state,n_pose = env.reset()
                changed_pose = list(n_pose)
                flag_first = False
                print("Reset")
                print(changed_pose)
            else:
                print("Reset pose")
                print(changed_pose)
                state,_ = env.reset_pose(changed_pose)
                rospy.sleep(5)


            print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end='\n')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0

            episode_reward = 0
            done = False
            i = 0
            score = 0
            while not done and i < STEPS_PER_EPISODE and not rospy.is_shutdown():
                i += 1
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                next_state, reward, done, _ = env.step(action)

                score += reward

                if not evaluation_episode:
                    print("TRAINING")
                    agent.train_on_transition(state, action, next_state, reward, done)
                else:
                    episode_reward += reward
                state = next_state
            if evaluation_episode:
                run_results.append(episode_reward)

            episode_number += 1

            print('EP Reward = %.01f' % score)
        agent_results.append(run_results)
        agent.save_net()

    # env.close()

    agent.save_net()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

    ax = plt.gca()
    ax.set_ylim([0, max(results_mean)])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    plt.legend(loc='best')
    plt.show()
