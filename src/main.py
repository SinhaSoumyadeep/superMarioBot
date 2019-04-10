import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from collections import Counter
from statistics import mean,median

LR = 1e-3

# env = gym.make('CartPole-v0')
# env.reset()
# goal_steps = 500
# score_requirement = 50
# initial_games = 10000

# mario env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env.reset()
goal_steps = 1500
score_requirement = 200
initial_games = 50





def random_games():
    for episode in range(5):
        print("the game we are playing is: ",episode)
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, rewards, done, info = env.step(action)
            if done:
                break






def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

# game
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            # print("the action taking place is: ",action)
            observation, rewards, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])


            prev_observation = observation

            score += rewards

            if done:
                break




        if score >= score_requirement:
            accepted_scores.append(score)

            # print("the values in game memory is", game_memory[0])
            # print ("hehehehhe: ", game_memory[0][1])


            for data in game_memory:
                # print("checking whats in data 1", data[1])
                if data[1] == 0:
                    output = [1, 0, 0, 0, 0, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0, 0, 0, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0, 0, 0, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1, 0, 0, 0]
                elif data[1] == 4:
                    output = [0, 0, 0, 0, 1, 0, 0]
                elif data[1] == 5:
                    output = [0, 0, 0, 0, 0, 1, 0]
                elif data[1] == 6:
                    output = [0, 0, 0, 0, 0, 0, 1]


                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)


    print('Avergae accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print (Counter(accepted_scores))

    return training_data



initial_population()












