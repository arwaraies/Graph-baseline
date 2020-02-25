from multiprocessing import Pool
import networkx as nx
import numpy as np
from itertools import product
import time

from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index, preferential_attachment, \
    cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

from random import choice

def make_graph(edges):
    G = nx.Graph()
    
    for edge in edges:
        row = edge.strip().split('\t')
        nodes= row[2].split('_')
        
        G.add_node(row[0], node_type=nodes[0])
        G.add_node(row[1], node_type=nodes[1])
        G.add_edge(row[0],row[1], edge_type=row[2])

    return G


def read_test_data(edges):
    test_edges = []
    
    for edge in edges:
        row = edge.strip().split('\t')        
        test_edges.append((row[0],row[1]))
        
    return test_edges


def sample_negative_edges(G, test_edges):
    neg_edges = []
    
    node_types = nx.get_node_attributes(G,'node_type')
    unique_node_types = set(node_types.values())
    
    all_possible_nodes = {}
    
    for node in unique_node_types:
        all_possible_nodes[node] = {k for k,v in node_types.items() if v == node}

    for edge in test_edges:
        lhs = edge[0]
        rhs = edge[1]
        rhs_type = node_types[rhs]

        possible_nodes = list(all_possible_nodes[rhs_type] - {G.neighbors(lhs)} - {lhs})
        
        if len(possible_nodes) > 0:
            new_node = choice(possible_nodes)
            neg_edges.append((lhs,new_node))
            G.add_edge(lhs,new_node)
           
        #new_node = choice(possible_nodes)
        #sample_size = len(possible_nodes)
        #count = 0
        
        #while node_types[new_node] != rhs_type and count < sample_size:
        #    new_node = choice(possible_nodes)
        #    count+=1
            
        #neg_edges.append((lhs,new_node))
        
        #G.add_edge(lhs,new_node)

    G.remove_edges_from(neg_edges)
    
    return neg_edges


def sample_negative_edges_other_ideas(G, test_edges):
    neg_edges = set()
    
    node_types = nx.get_node_attributes(G,'node_type')
    unique_node_types = set(node_types.values())
    
    all_possible_nodes = {}
    
    for node in unique_node_types:
        all_possible_nodes[node] = {k for k,v in node_types.items() if v == node}
        
    
    all_lhs = {k for k,v in test_edges}
    
    all_neighbors = {}
    
    for lhs in all_lhs:
        all_neighbors[lhs] = {G.neighbors(lhs)}
    
    for edge in test_edges:
        lhs = edge[0]
        rhs = edge[1]
        rhs_type = node_types[rhs]
        
        #possible_nodes = all_possible_nodes[rhs_type] - {G.neighbors(lhs)} - {lhs}
        possible_nodes = all_possible_nodes[rhs_type] - all_neighbors[lhs] - {lhs}
        #possible_nodes = list(all_possible_nodes[rhs_type] - {G.neighbors(lhs)} - {lhs})
        #possible_nodes = list(all_possible_nodes[rhs_type] - all_neighbors[lhs] - {lhs})
        
        if len(possible_nodes) > 0:
            new_node = choice(tuple(possible_nodes))
            neg_edges.add((lhs,new_node))
            G.add_edge(lhs,new_node)
           
        #new_node = choice(possible_nodes)
        #sample_size = len(possible_nodes)
        #count = 0
        
        #while node_types[new_node] != rhs_type and count < sample_size:
        #    new_node = choice(possible_nodes)
        #    count+=1
            
        #neg_edges.append((lhs,new_node))
        
        #G.add_edge(lhs,new_node)

    G.remove_edges_from(neg_edges)
    
    return neg_edges


def sample_negative_edges_old(G, test_edges):
    neg_edges = []
    
    node_types = nx.get_node_attributes(G,'node_type')
    
    for edge in test_edges:
        lhs = edge[0]
        rhs = edge[1]
        rhs_type = node_types[rhs]
        
        possible_nodes = list({k for k,v in node_types.items() if v == rhs_type} - {G.neighbors(lhs)} - {lhs})
        
        if len(possible_nodes) > 0:
            new_node = choice (possible_nodes)
            neg_edges.append((lhs,new_node))
            G.add_edge(lhs,new_node)
           
        #new_node = choice(possible_nodes)
        #sample_size = len(possible_nodes)
        #count = 0
        
        #while node_types[new_node] != rhs_type and count < sample_size:
        #    new_node = choice(possible_nodes)
        #    count+=1
            
        #neg_edges.append((lhs,new_node))
        
        #G.add_edge(lhs,new_node)

    G.remove_edges_from(neg_edges)
    
    return neg_edges


def calculate_scores(tuples):
    test_tuples = tuples[0]
    negs_tuples = tuples[1]
    model = tuples[2]
    print(model)
    pred_scores = []
    true_scores = []
        
    for u, v, p in test_tuples:
        pred_scores.append(p)
        true_scores.append(1.0)
        
    for u, v, p in negs_tuples:
        pred_scores.append(p)
        true_scores.append(0.0)

    #calculate scores
    aps = average_precision_score(true_scores,pred_scores)
    roc = roc_auc_score(true_scores,pred_scores)
    precision, recall, _ = precision_recall_curve(true_scores,pred_scores)
    
    aupr = auc(recall,precision)
    fpr, tpr, _ = roc_curve(true_scores,pred_scores)
    
    results = model + '\t' + str(aps) + '\t' + str(roc) + '\t' + str(aupr) + '\n'
    
    return [results, recall, precision, fpr, tpr, model]


def calculate_scores_star(a_b_c):
    return calculate_scores(*a_b_c)


def find_target_edges(G,edges):
    target_edges = []
    
    node_types = nx.get_node_attributes(G,'node_type')
    
    for edge in edges:
        if (node_types[edge[0]] == 'DISEASE' and node_types[edge[1]] == 'GENE') or (node_types[edge[0]] == 'GENE' and node_types[edge[1]] == 'DISEASE'):
            target_edges.append((edge[0],edge[1]))

    return target_edges


def find_target_edges_star(a_b):
    return find_target_edges(*a_b)

print('read data')
#read the data
fin = open('v03/valid.txt','r')
lines = fin.readlines()
fin.close()

fin = open('v03/test.txt','r')
lines = lines + fin.readlines()
fin.close()

#read the testing data
fin = open('v03/test.txt','r')
test_lines = fin.readlines()
test_edges = read_test_data(test_lines)
fin.close()

print('generate training graph')
#generate the training graph
numberOfThreads = 2
pool = Pool(processes=numberOfThreads)
chunks = np.array_split(lines, numberOfThreads)
graphs = pool.imap_unordered(make_graph,chunks)
pool.close()
pool.join()

G = nx.compose_all(graphs)

print('sample negative edges')
#sample negative edges
start = time.time()
neg_edges = sample_negative_edges(G, test_edges[0:1000])
end = time.time()
print(end-start)

print(len(neg_edges))
print(len(test_edges))
G.remove_edges_from(test_edges)

print('generate the models')
#calculate the scores for all testing edges
testing_tuples = [resource_allocation_index(G,test_edges[0:1000]), jaccard_coefficient(G,test_edges[0:1000]), \
                 preferential_attachment(G,test_edges[0:1000])]
#testing_tuples = [resource_allocation_index(G,test_edges[0:1000])]

#calculate the scores for all non-existing edges
neg_tuples = [resource_allocation_index(G,neg_edges), jaccard_coefficient(G,neg_edges), \
                 preferential_attachment(G,neg_edges)]
#neg_tuples = [resource_allocation_index(G,neg_edges)]

#list of methods
models = ['resource_allocation_index', 'jaccard_coefficient', 'preferential_attachment']
#models = ['resource_allocation_index']

fout = open('baseline_performance.txt','w')
fout.write('Method\tAverage Precision Score\tAUROC\tAUPR\n')

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

plt.figure(1, figsize=(10,12))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')

plt.figure(2, figsize=(10,12))    
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Reciver Operator Curve')

count = 0
print('calculate the scores')

for output in map(calculate_scores,zip(testing_tuples,neg_tuples,models)):
    results, recall, precision, fpr, tpr, model = output
    
    fout.write(results)
    plt.figure(1)
    plt.plot(recall, precision, color=color_sequence[count], label=model)
    
    plt.figure(2)
    plt.plot(fpr, tpr, color=color_sequence[count], label=model)
    
    count+=1
    
    
plt.figure(1)
plt.legend(loc='lower center',ncol=1, bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig('PR.png', bbox_inches="tight")
plt.close()

plt.figure(2)
plt.legend(loc='lower center',ncol=1, bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig('ROC.png', bbox_inches="tight")
plt.close()

fout.close()

#calcuate the scores for target and disease edges only
print('calcuate the scores for targets and diseases only')

print('find target and disease edges')

target_test_edges = find_target_edges(G,test_edges)
print('sample negative edges')
#sample negative edges
G.add_edges_from(target_test_edges)
target_neg_edges = sample_negative_edges(G, target_test_edges[0:1000])
print(len(target_neg_edges))
print(len(target_test_edges))
G.remove_edges_from(target_test_edges)

print('generate the models')
#calculate the scores for all testing edges
testing_tuples = [resource_allocation_index(G,target_test_edges), jaccard_coefficient(G,target_test_edges), \
                 preferential_attachment(G,target_test_edges[0:1000])]
#testing_tuples = [resource_allocation_index(G,test_edges[0:1000])]

#calculate the scores for all non-existing edges
neg_tuples = [resource_allocation_index(G,target_neg_edges), jaccard_coefficient(G,target_neg_edges), \
                 preferential_attachment(G,target_neg_edges)]
#neg_tuples = [resource_allocation_index(G,neg_edges)]

#list of methods
models = ['resource_allocation_index', 'jaccard_coefficient', 'preferential_attachment']
#models = ['resource_allocation_index']

fout = open('baseline_performance_targets.txt','w')
fout.write('Method\tAverage Precision Score\tAUROC\tAUPR\n')

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

plt.figure(1, figsize=(10,11))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')

plt.figure(2, figsize=(10,11))    
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Reciver Operator Curve')

count = 0
print('calculate the scores')
for output in map(calculate_scores,zip(testing_tuples,neg_tuples,models)):
    results, recall, precision, fpr, tpr, model = output
    
    fout.write(results)
    plt.figure(1)
    plt.plot(recall, precision, color=color_sequence[count], label=model)
    
    plt.figure(2)
    plt.plot(fpr, tpr, color=color_sequence[count],label=model)
    
    count+=1
    
    
plt.figure(1)
plt.legend(loc='lower center',ncol=1, bbox_to_anchor=(0.5, -0.23), frameon=False)
plt.tight_layout()
plt.savefig('PR_targets.png', bbox_inches="tight")
plt.close()

plt.figure(2)
plt.legend(loc='lower center',ncol=1, bbox_to_anchor=(0.5, -0.23), frameon=False)
plt.tight_layout()
plt.savefig('ROC_targets.png', bbox_inches="tight")
plt.close()

fout.close()

