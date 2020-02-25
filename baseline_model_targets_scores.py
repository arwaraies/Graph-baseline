import networkx as nx

from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index, preferential_attachment, \
    cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster

import random as ra
from statistics import mean
from itertools import product
import numpy as np
from multiprocessing import Process, Array
from math import ceil

def make_graph(edges):
    G = nx.Graph()
    
    for edge in edges:
        row = edge.strip().split('\t')
        nodes= row[2].split('_')
        
        if nodes[0] == 'DISEASE':
            G.add_node(row[0], node_type=nodes[0])
        elif row[0] not in G.nodes():
            G.add_node(row[0], node_type=nodes[0])
        
        
        if nodes[1] == 'DISEASE':
            G.add_node(row[1], node_type=nodes[1])
        elif row[1] not in G.nodes():
            G.add_node(row[1], node_type=nodes[1])
            
        if (nodes[0] == 'GENE' and nodes[1] == 'DISEASE') or (nodes[0] == 'DISEASE' and nodes[1] == 'GENE'):
            G.add_edge(row[0],row[1], edge_type=row[2], is_target_disease=True)
        else:
            G.add_edge(row[0],row[1], edge_type=row[2], is_target_disease=False)
        
    return G


def takeSecond(elem):
    return elem[1]


def takeThird(elem):
    return elem[2]


def rank_disease_edges(G,u,train_diseases, model):
    if model == 'cn_soundarajan_hopcroft' or model == 'ra_index_soundarajan_hopcroft':
        model+= '(G,list(product([u],train_diseases)),\'node_type\')'
    else:
        model+= '(G,list(product([u],train_diseases)))'
    
    scores_tuples = eval(model)
    sorted_tuples = sorted(scores_tuples, key=takeThird, reverse=True)
    
    return [b for a,b,c in sorted_tuples]


def rank_targets_edges(G,v,train_targets, model):
    if model == 'cn_soundarajan_hopcroft' or model == 'ra_index_soundarajan_hopcroft':
        model+= '(G,list(product(train_targets,[v])),\'node_type\')'
    else:
        model+= '(G,list(product(train_targets,[v])))'
    
    scores_tuples = eval(model)
    sorted_tuples = sorted(scores_tuples, key=takeThird, reverse=True)
    
    return [a for a,b,c in sorted_tuples]


def calculate_scores(G,testing_tuples,train_targets,train_diseases,model):
    size = len(testing_tuples)
    rank = Array('d', np.empty(size*2))
    reciprocal_rank = Array('d', np.empty(size*2))
    r1 = Array('d', np.empty(size*2))
    r10 = Array('d', np.empty(size*2))
    r50 = Array('d', np.empty(size*2))
    sorted_tuples_genes = sorted(testing_tuples)
    sorted_tuples_diseases = sorted(target_test_edges, key=takeSecond)

    def taregt_function(rank,reciprocal_rank,r1,r10,r50,sorted_tuples_genes,index,stride):
        gene = ''
        count = index
        
        if index+stride <= size:
            max_range = index+stride
        else:
            max_range = size
        
        for i in range(index,max_range):
            u, v = sorted_tuples_genes[i]
            
            if count%100 == 0:
                print(str(count))
            
            count+=1
            
            if u != gene:
                ranked_edges = rank_disease_edges(G,u,train_diseases, model)
                gene = u
                
            r = ranked_edges.index(v) + 1
            rank[i] = r
            reciprocal_rank[i] = 1.0/r
            
            if r == 1:
                r1[i]=1.0
            else:
                r1[i]=0.0
            
            if r <= 10:
                r10[i]=1.0
            else:
                r10[i]=0.0
                
            if r <= 50:
                r50[i]=1.0
            else:
                r50[i]=0.0
            
    def disease_function(rank,reciprocal_rank,r1,r10,r50,sorted_tuples_diseases,index,stride):
        disease = ''
        count = index
        
        if index+stride <= size:
            max_range = index+stride
        else:
            max_range = size
            
        for i in range(index,max_range):
            u, v = sorted_tuples_diseases[i]
            
            if count%100 == 0:
                print(str(count))
            
            count+=1
            
            if v != disease:
                ranked_edges = rank_targets_edges(G,v,train_targets, model)
                disease = v
                
            r = ranked_edges.index(u) + 1
            rank[i+size] = r
            reciprocal_rank[i+size] = 1.0/r
            
            if r == 1:
                r1[i+size]=1.0
            else:
                r1[i+size]=0.0
            
            if r <= 10:
                r10[i+size]=1.0
            else:
                r10[i+size]=0.0
                
            if r <= 50:
                r50[i+size]=1.0
            else:
                r50[i+size]=0.0
                
    num_processes = 5
    stride = ceil(size/num_processes)
    
    gene_processes = []
    disease_processes = []
    
    for i in range(num_processes):
        gene_processes.append(Process(target=taregt_function,args=(rank,reciprocal_rank,r1,r10,r50,sorted_tuples_genes,i*stride,stride)))
        disease_processes.append(Process(target=disease_function,args=(rank,reciprocal_rank,r1,r10,r50,sorted_tuples_diseases,i*stride,stride)))

    for i in range(num_processes):
        gene_processes[i].start()
        
    for i in range(num_processes):
        disease_processes[i].start()

    for i in range(num_processes):
        gene_processes[i].join()
        
    for i in range(num_processes):
        disease_processes[i].join()
    
    fout = open(model+'.txt','w')
    fout.write('Method\tMean Rank\tMRR\tHit@1\tHit@10\tHit@50\n')
    fout.write(model+'\t'+str(mean(rank))+'\t'+str(mean(reciprocal_rank))+'\t'+\
               str(mean(r1))+'\t'+str(mean(r10))+'\t'+str(mean(r50))+'\n')
    fout.close()
    
    #return [mean(rank), mean(reciprocal_rank), mean(r1), mean(r10), mean(r50)]
    


print('read data')
#read the data
fin = open('v03/test.txt','r')
lines = fin.readlines()
fin.close()

print('generate training graph')
G=make_graph(lines)

print('find target edges')
target_edges = [(u,v) if G.node[u]['node_type'] == 'GENE' else (v,u) for u,v,d in G.edges(data=True) if d['is_target_disease'] == True]
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
targets = [n for n,d in G.nodes(data=True) if d['node_type'] == 'GENE']
diseases = [n for n,d in G.nodes(data=True) if d['node_type'] == 'DISEASE']

print('Number of Targets: '+str(len(targets)))
print('Number of diseases: '+str(len(diseases)))
print('Number of existing target disease associations: '+str(len(target_edges)))
print('Number of non-existing target disease associations: '+str((len(targets)*len(diseases))-len(target_edges)))

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
train_targets = [n for n,d in G_train.nodes(data=True) if d['node_type'] == 'GENE']
train_diseases = [n for n,d in G_train.nodes(data=True) if d['node_type'] == 'DISEASE']
print('Number of targets in the training set: '+str(len(train_targets)))
print('Number of diseases in the training set: '+str(len(train_diseases)))

print('generate the models')
G.remove_edges_from(target_test_edges)

#list of methods
models = ['cn_soundarajan_hopcroft', 'ra_index_soundarajan_hopcroft', 'adamic_adar_index', 'resource_allocation_index', \
          'jaccard_coefficient', 'preferential_attachment']

print('calculate the scores')

for model in models:
    Process(target = calculate_scores, args=(G,target_test_edges,targets,diseases,model)).start()



