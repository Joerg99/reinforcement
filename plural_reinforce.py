import gym
import math
import numpy as np
import random

env = gym.make('CartPole-v0')
env.reset()
# for i_episode in range(20):
#     print(' ------ EPISODE ', i_episode, ' -------- ')
#     observation = env.reset()
#     for t in range(300):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(reward)
#         if done:
#             print('ITS OVER!!!!!!!!! STOPT IT!')
#             break
#     

NUM_BUCKETS = (1,1,6,3) # discretize state space (cart position(ignore), cart velocity(ignore), angular pos of pole (6 possible values), angular velocity (3 possible values))
NUM_ACTIONS = env.action_space.n

STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,)) # shape: 1,1,6,3,2



EXPLORE_RATE_MIN = 0.01
LEARNING_RATE_MIN = 0.1



def get_explore_rate(t):
    return max(EXPLORE_RATE_MIN, min(1 , 1.0 - math.log10( (t+1)/25 )))

def get_learning_rate(t):
    return max(LEARNING_RATE_MIN, min(0.5 , 1.0 - math.log10( (t+1)/25 )))


def select_action(state, explore_rate): # either random or action with max reward
    if random.random() < explore_rate:
        action = env.action_space.sample() # random action
    else: # max action
        action = np.argmax(q_table[state])
    return action

def state_to_bucket(state):
    bucket_indices = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] -1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            
            offset = (NUM_BUCKETS[i]-1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i]-1) / bound_width
            
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)


def simulate():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99
    num_streaks = 0
    
    for episode in range(20):
        observation = env.reset()
        
        state_0 = state_to_bucket(observation)
        
        for t in range(250):
            env.render()
            action = select_action(state_0, explore_rate)
            observation, reward, done, info = env.step(action) 
            state = state_to_bucket(observation)
            best_q = np.amax(q_table[state])
            
            q_table[state_0 + (action,)] += learning_rate *(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])
            state_0 = state
            
            if done:
                print("episode over: ", episode, t)
                if t >249:
                    num_streaks +=1
                else:
                    num_streaks = 0
                break
        print(q_table)
        if num_streaks > 100:
            break
        explore_rate= get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

simulate()
env.close()