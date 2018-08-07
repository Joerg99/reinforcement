import pylab as plt
import networkx as nx
import numpy as np

# Database of questions
questions = [['q101', 'q102', 'q103', 'q104'], \
             ['q201', 'q202', 'q203', 'q204', 'q205', 'q206', 'q207', 'q208', 'q209', 'q210'], \
             ['q301', 'q302']]


edges = [(0,101,0), (0,102,0), (0,103,0), (0,103,0), (0,104,0), \
         (101,201,0), (101,202,0), (101,203,0), (102,204,0), (102,205,0), (103,206,0),(103,207,0), (103,208,0), (104,209,0), (104,210,0), \
         (202, 301,0), (202,302,0)]


G = nx.Graph()
G.add_weighted_edges_from(edges)
position = nx.spring_layout(G)
 
nx.draw_networkx_nodes(G, position)
nx.draw_networkx_edges(G, position)
nx.draw_networkx_labels(G, position)

# G[0][102]['weight'] += 100
# G[0][102]['weight'] += 77
# print(G[0][102])

#plt.show()     


def pick_question(index):
    question = np.random.choice(questions[index])
    # question = n
    return question

def update_graph(edge_from, edge_to):
    try:
        G[edge_from][edge_to]['weight'] += 1
        return True
    except:
        print('gibts nicht :(')
        return False
        


def answer_options(index):
    eds = G.edges(data=True)
    possible_answers = []
    for e in eds:
        if e[0] == index:
            possible_answers.append(e[1])
    return possible_answers

for _ in range(10):
    index= 0
    while(index < 3): # index < tree depth
        print(pick_question(index))
        
        if index == 0: # 1st question is a special case
            if len(answer_options(0)) != 0:
                print('choose: ', answer_options(0))
                answer_current = input('Input: ')
                answer_available = update_graph(index, int(answer_current))
                if answer_available:
                    index +=1
            else:
                print('~~~~~ Blatt erreicht ~~~~~')
                break
        else: # previous answer and current answer needed to address the correct edge
            answer_prev = answer_current
            if len(answer_options(int(answer_prev))) != 0:
                print('choose: ', answer_options(int(answer_prev)))
                answer_current = input('Input: ')
                answer_available = update_graph(int(answer_prev), int(answer_current))
                if answer_available:
                    index += 1
            else:
                print('~~~~~ Blatt erreicht ~~~~~')
                break #index = 0
    
    # loop for testing
#     if index == 3:
eds = G.edges(data=True)
for e in eds:
    print(e)
#         index = 0
