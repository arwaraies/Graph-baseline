# -*- coding: utf-8 -*-

from multiprocessing import Pool
import networkx as nx
import numpy as np
from itertools import product

from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index, preferential_attachment, \
    cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

import random as ra
import time
import csv

def make_graph(edges):
    G = nx.Graph()
    
    for edge in edges:
        row = edge.strip().split('\t')
        nodes= row[2].split('_')
    
        G.add_node(row[0], node_type=nodes[0])
        G.add_node(row[1], node_type=nodes[1])

        if (nodes[0] == 'GENE' and nodes[1] == 'DISEASE') or (nodes[0] == 'DISEASE' and nodes[1] == 'GENE'):
            G.add_edge(row[0],row[1], edge_type=row[2], is_target_disease=True)
        else:
            G.add_edge(row[0],row[1], edge_type=row[2], is_target_disease=False)
        
    return G


def read_test_data(edges):
    test_edges = []
    target_test_edges = []

    for edge in edges:
        row = edge.strip().split('\t')      
        test_edges.append((row[0],row[1]))
        
        types = row[2].split('_')

        if (types[0] == 'DISEASE' and types[1] == 'GENE') or (types[0] == 'GENE' and types[1] == 'DISEASE'):
            target_test_edges.append((row[0],row[1]))

    return test_edges, target_test_edges


def sample_negative_edges(G, test_edges, n):
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
        
        #possible_nodes = set(G.nodes())
        possible_nodes = list(all_possible_nodes[rhs_type] - {G.neighbors(lhs)} - {lhs})
        
        if len(possible_nodes) > n:
            ra.seed(0)
            new_sample = ra.sample(possible_nodes,n)
            
            for new_node in new_sample:
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


def calculate_imbalanced_scores(tuples):
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
    
    
    return [aps, roc, aupr, model]


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


def get_type_id(lines):
    type2id = {}
    count = 0

    for line in lines:
        row = line.strip().split('\t')
        types = row[2].split('_')
        
        if types[0] not in type2id.keys():
            type2id[types[0]] = count
            count+=1

        if types[1] not in type2id.keys():
            type2id[types[1]] = count
            count+=1

    return type2id


print('read data')

#read the data
fin = open('v03/full.txt','r')
lines = fin.readlines()
fin.close()

print('find target edges')
target_edges = [(u,v) for u,v,d in G.edges(data=True) if d['is_target_disease'] == True]
size=int((len(target_edges)/100)*20)
ra.seed(0)
target_test_edges = ra.sample(target_edges,size)

print('graph statistics')
print('Number of nodes: '+str(nx.number_of_nodes(G)))
node_types = len(set(d['node_type'] for n,d in G.nodes(data=True)))
print('Number of node types: '+str(node_types))
print('Number of edges: '+str(nx.number_of_edges(G)))
fin = open('v03/dynamic_rel_count.txt','r')
print('Number of edges types: '+fin.readline())
fin.close()
targets = len([n for n,d in G.nodes(data=True) if d['node_type'] == 'GENE'])
diseases = len([n for n,d in G.nodes(data=True) if d['node_type'] == 'DISEASE'])
print('Number of Targets: '+str(targets))
print('Number of diseases: '+str(diseases))
print('Number of existing target disease associations: '+str(len(target_edges)))
print('Number of non-existing target disease associations: '+str((targets*diseases)-len(target_edges)))

print('Targets disease associations in the testing set: '+ str(len(target_test_edges)))
G_test = G.edge_subgraph(target_test_edges)
test_targets = len([n for n,d in G_test.nodes(data=True) if d['node_type'] == 'GENE'])
test_diseases = len([n for n,d in G_test.nodes(data=True) if d['node_type'] == 'DISEASE'])
print('Number of targets in the testing set: '+str(test_targets))
print('Number of diseases in the testing set: '+str(test_diseases))

print('Number of edges in training set: '+str((nx.number_of_edges(G))-len(target_test_edges)))
print('Number of target diseases edges in the training set: '+str(len(target_edges)-len(target_test_edges)))
print('Number of edges between other entities in the training set: '+str(nx.number_of_edges(G)-len(target_edges)))
target_train_edges = set(G.edges)
target_train_edges.difference_update(target_test_edges)
G_train = G.edge_subgraph(list(target_train_edges))
train_targets = len([n for n,d in G_train.nodes(data=True) if d['node_type'] == 'GENE'])
train_diseases = len([n for n,d in G_train.nodes(data=True) if d['node_type'] == 'DISEASE'])
print('Number of targets in the training set: '+str(train_targets))
print('Number of diseases in the training set: '+str(train_diseases))

#fout.close()

#calcuate the scores for target and disease edges only
print('calcuate the scores for targets and diseases only')

print('sample negative edges')
#sample negative edges
#G.add_edges_from(target_test_edges)
target_neg_edges = sample_negative_edges(G, target_test_edges,1)
print(len(target_neg_edges))
print(len(target_test_edges))
G.remove_edges_from(target_test_edges)

print('generate the models')
#calculate the scores for all testing edges
testing_tuples = [cn_soundarajan_hopcroft(G,target_test_edges,'node_type'), ra_index_soundarajan_hopcroft(G,target_test_edges,'node_type'), adamic_adar_index(G,target_test_edges), resource_allocation_index(G,target_test_edges), jaccard_coefficient(G,target_test_edges), preferential_attachment(G,target_test_edges)]
#testing_tuples = [resource_allocation_index(G,test_edges[0:1000])]

#calculate the scores for all non-existing edges
neg_tuples = [cn_soundarajan_hopcroft(G,target_neg_edges,'node_type'), ra_index_soundarajan_hopcroft(G,target_neg_edges,'node_type'), adamic_adar_index(G,target_neg_edges), resource_allocation_index(G,target_neg_edges), jaccard_coefficient(G,target_neg_edges), preferential_attachment(G,target_neg_edges)]
#neg_tuples = [resource_allocation_index(G,neg_edges)]

#list of methods
models = ['cn_soundarajan_hopcroft', 'ra_index_soundarajan_hopcroft', 'adamic_adar_index', 'resource_allocation_index', 'jaccard_coefficient', 'preferential_attachment']
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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
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
plt.legend(loc='lower center',ncol=2, bbox_to_anchor=(0.5, -0.23), frameon=False)
plt.tight_layout()
plt.savefig('PR_targets.png', bbox_inches="tight")
plt.close()

plt.figure(2)
plt.legend(loc='lower center',ncol=2, bbox_to_anchor=(0.5, -0.23), frameon=False)
plt.tight_layout()
plt.savefig('ROC_targets.png', bbox_inches="tight")
plt.close()

fout.close()


print('calcuate the socres for imbalanced ratio')

fout = open('baseline_performance_targets_imbalanced.txt','w')
fout.write('Method\tImbalanced Ratio\tAverage Precision Score\tAUROC\tAUPR\n')

plt.figure(3, figsize=(10,11))
plt.xlabel('Imbalanced Ratio')
plt.ylabel('AUPRC')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')

plt.figure(4, figsize=(10,11))    
plt.xlabel('Imbalanced Ratio')
plt.ylabel('AUROC')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Reciver Operator Curve')

plt.figure(5, figsize=(10,11))    
plt.xlabel('Imbalanced Ratio')
plt.ylabel('Average Precision Score')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Average Precision Score')

for im in [1, 10, 50, 100, 500]:
    print('imbalance ratio: '+str(im))

    print('sample negative edges')
    G.add_edges_from(target_test_edges)
    target_neg_edges = sample_negative_edges(G,target_test_edges,im)
    print(len(target_test_edges))
    print(len(target_neg_edges))
    G.remove_edges_from(target_test_edges)

    print('generate the models')
    neg_tuples = [cn_soundarajan_hopcroft(G,target_neg_edges,'node_type'), ra_index_soundarajan_hopcroft(G,target_neg_edges,'node_type'), adamic_adar_index(G,target_neg_edges), resource_allocation_index(G,target_neg_edges), jaccard_coefficient(G,target_neg_edges), preferential_attachment(G,target_neg_edges)]
    
    count = 0
    print('calculate the scores')
    
    for output in map(calculate_scores,zip(testing_tuples,neg_tuples,models)):
        apr, roc, aupr, model = output
        
        fout.write(model+'\t'+str(im)+'\t'+results+'\n')
        plt.figure(3)
        plt.plot(im, precision, color=color_sequence[count], label=model)
        
        plt.figure(4)
        plt.plot(fpr, tpr, color=color_sequence[count],label=model)
        
        count+=1
