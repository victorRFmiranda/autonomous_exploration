import os
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve

from environment.environment import Environment
import action_mapper



if __name__ == '__main__':
    # env = gym.make('InvertedPendulumBulletEnv-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('Pendulum-v1')

    env = Environment("./environment/world/four_rooms")
    env.use_ditance_angle_to_end(True)
    env.set_observation_rotation_size(128)
    env.use_observation_rotation_size(True)
    env.set_cluster_size(1)

    obs,_,_,_ = env.reset()

    action_space = 2
    agent = Agent(input_dims=obs.shape, env=env,
            n_actions=action_space)
    # agent = Agent(input_dims=env.observation_space.shape, env=env,
    #         n_actions=env.action_space.shape[0])
    n_games = 5000
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = 100 
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    for i in range(n_games):
        # if (i % 300 == 0):
        #   os.system("pkill -x gnuplot_qt")
        #   lista = ["./environment/world/ufmg_2", "./environment/world/room", "./environment/world/four_rooms", "./environment/world/roblab"]
        #   env = Environment(np.random.choice(lista))
        #   env.use_ditance_angle_to_end(True)
        #   env.set_observation_rotation_size(128)
        #   env.use_observation_rotation_size(True)
        #   env.set_cluster_size(1)

        observation,_,_,_ = env.reset()
        done = False
        score = 0
        durantion = 0
        while not done:
            env.visualize()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(float(action[0]),float(action[1]),20)
            # penality for stand still
            if(durantion > 500):
                reward = -40
                done = True
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            durantion+=1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'duration %d' % durantion, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    agent.save_models()

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

