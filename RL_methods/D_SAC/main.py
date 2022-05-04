from environment.environment import Environment
import action_mapper
import numpy as np
import matplotlib.pyplot as plt
import os

from Discrete_SAC_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 4
RUNS = 2
EPISODES_PER_RUN = 1000
STEPS_PER_EPISODE = 300
LOAD_MODELS = False

if __name__ == "__main__":
    env = Environment("./environment/world/four_rooms")
    env.use_ditance_angle_to_end(True)
    env.set_cluster_size(1)
    obs,_,_,_ = env.reset()

    action_space = action_mapper.ACTION_SIZE

    agent_results = []
    for run in range(RUNS):
        # os.system("pkill -x gnuplot_qt")
        # lista = ["./environment/world/room", "./environment/world/four_rooms", "./environment/world/ufmg_2", "./environment/world/roblab"]
        # env = Environment(lista[run])
        # env.use_ditance_angle_to_end(True)
        # env.set_observation_rotation_size(128)
        # env.use_observation_rotation_size(True)
        # env.set_cluster_size(1)

        # agent = SACAgent(obs.shape,action_space)
        agent = SACAgent(obs.shape,action_space)
        if LOAD_MODELS:
            print("Loading Models")
            agent.load_net()

        run_results = []
        for episode_number in range(EPISODES_PER_RUN):
            print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            # evaluation_episode = True
            episode_reward = 0
            state,_,_,_ = env.reset()
            # state = env.reset()
            done = False
            i = 0
            while not done and i < STEPS_PER_EPISODE:
                env.visualize()
                i += 1
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                linear, angular = action_mapper.map_action(action)
                next_state, reward, done, info = env.step(linear, angular, 20)
                # next_state, reward, done, info = env.step(action)
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                else:
                    episode_reward += reward
                state = next_state
            if evaluation_episode:
                run_results.append(episode_reward)

            # print('Reward = %.01f' % episode_reward)
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
