import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1 #alpha
DISCOUNT_FACTOR = .99 #gamma (y)
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        while not done:
            # e-greedy algorithm
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Q-learning
            prev_value = Q_table[(obs, action)]
            next_max = np.max([Q_table[(next_obs, i)] for i in range(env.action_space.n)])

            # Update Q-value for current state
            if not done:
                new_value = (1 - LEARNING_RATE) * prev_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
            else:
                new_value = (1 - LEARNING_RATE) * prev_value + LEARNING_RATE * reward

            Q_table[(obs, action)] = new_value

            # Move to the next state
            obs = next_obs

            episode_reward += reward

            # Decay epsilon
        EPSILON *= EPSILON_DECAY

        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################