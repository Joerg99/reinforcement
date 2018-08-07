import pylab as plt
import networkx as nx
import numpy as np

questions = [['q101', 'q102', 'q103', 'q104'], \
             ['q201', 'q202', 'q203', 'q204', 'q205', 'q206', 'q207', 'q208', 'q209', 'q210'], \
             ['q301', 'q302']]


edges = [(0,101), (0,102), (0,103), (0,103), (0,104), \
         (101,201), (101,202), (101,203), (102,204), (102,205), (103,206),(103,207), (103,208), (104,209), (104,210), \
         (202, 301), (202,302)]


G = nx.Graph()
G.add_edges_from(edges)
position = nx.spring_layout(G)
 
nx.draw_networkx_nodes(G, position)
nx.draw_networkx_edges(G, position)
nx.draw_networkx_labels(G, position)
#plt.show()     


def setup_reward_matrix(edges, num_layers):
    rewards = []
    for i in range(num_layers):
        count = 0
        for edge in edges:
            if int(str(edge[0])[0]) == i:
                count += 1
        rewards.append([0]*count)
    return rewards
    
q_table = setup_reward_matrix(edges, 3)

def pick_question(index):
    question_index = q_table.index(max(q_table))
    pos = [index, question_index]
    return questions[index][question_index], pos

def update_q_table(position):
    q_table[position[0]][position[1]] +=1

 
index= 0
while(index < 3):
    question, question_position = pick_question(index)
    print(question, 'y? n?')
    answer = input()
     
    if answer == 'y':
        index += 1
        update_q_table(question_position)
        if index == 3:
            index = 0
     
    print(q_table)



# 
# # SETUP REWARDS
# # Reward Matrix size == Number of nodes in graph
# # rewards = [ [a b c] [k l m] [x y z] ] lines = from, columns = to
# rewards = np.matrix(np.ones((size_matrices,size_matrices)))
# rewards = rewards * -1 # init all values -1
# # APPLY REWARDS TO EDGES --> edge is a tuple (from, to)
# # edges to goal 100, all other existing edges 0. Other entries -1
# 
# for edge in edges:
#     if edge[1] == goal:
#         rewards[edge] = 100
#     else:
#         rewards[edge] = 0
#     if edge[0] == goal:
#         rewards[edge[::-1]] = 100
#     else:
#         rewards[edge[::-1]]=0
# rewards[goal,goal] = 100
# 
# discount_factor = 0.8
# 
# #SETUP Q_TABLE
# 
# q_table = np.matrix(np.zeros((size_matrices,size_matrices)))
# pd.DataFrame(q_table)
# 
# ####################### state = number of node
# 
# def get_available_actions(state):
#     current_state_row = rewards[state,]
#     # available actions = adjacent nodes from current state
#     available_actions = np.where(current_state_row >=0)[1]
#     return available_actions
# 
# def sample_next_action(available_actions):
#     # select random action from available actions
#     return int(np.random.choice(available_actions, size=1))
# 
# def update(current_state, action, discount_factor):
#     max_index = np.where(q_table[action, ] == np.max(q_table[action, ]))[1] #find index of highest q-value for a certain action
#     
#     if max_index.shape[0] > 1:
#         max_index = int(np.random.choice(max_index, size=1))
#     else:
#         max_index = int(max_index)
# #     print(max_index)
#     max_value = q_table[action, max_index]
#     q_table[current_state, action] = rewards[current_state, action] + discount_factor * max_value
# 
# ################################
# # update q_table 1000 times
# # start from random node
# for _ in range(1000):
#     current_state = np.random.randint(0, int(q_table.shape[0]))
#     available_actions = get_available_actions(current_state)
#     action = sample_next_action(available_actions)
#     update(current_state, action, discount_factor)
# ################################
# 
# current_state = 5
# 
# 
# print('find a best way', current_state, ' to ', goal)
# 
# 
# #find best way from one node to another. best = max reward
# def find_way(start, goal):
#     steps = [start]
#     
#     current_state = start
#     while current_state != goal:
#         next_step_index = np.where(q_table[current_state,] == np.max(q_table[current_state, ]))[1]
#     
#         if next_step_index.shape[0] > 1:
#             next_step_index = int(np.random.choice(next_step_index, size = 1))
#         else:
#             next_step_index = int(next_step_index)
#         #print(next_step_index)
#         steps.append(next_step_index)
#         current_state = next_step_index
#     return steps
# 
# print("Most efficient: ", find_way(current_state, goal))


