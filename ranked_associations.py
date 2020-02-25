# -*- coding: utf-8 -*-

import networkx as nx

from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index, preferential_attachment, \
    cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster

import random as ra
from statistics import mean
from itertools import product
import numpy as np
from multiprocessing import Process, Array
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


def calculate_scores(G,target_edges,all_edges,model):
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
fin = open('v03/full.txt','r')
lines = fin.readlines()
fin.close()

print('generate training graph')
G=make_graph(lines)

print('find target edges')
target_edges = set((u,v) if G.node[u]['node_type'] == 'GENE' else (v,u) for u,v,d in G.edges(data=True) if d['is_target_disease'] == True)
targets = [n for n,d in G.nodes(data=True) if d['node_type'] == 'GENE']
diseases = [n for n,d in G.nodes(data=True) if d['node_type'] == 'DISEASE']
all_edges = list(product(targets,diseases))
#all_edges = all_edges[0:1000000]
#list of methods
#models = ['cn_soundarajan_hopcroft', 'ra_index_soundarajan_hopcroft', 'adamic_adar_index', 'resource_allocation_index', \
#          'jaccard_coefficient', 'preferential_attachment']

print('calculate the scores')
scores_tuples = resource_allocation_index(G,all_edges)

print('sort the scores')
sorted_tuples = sorted(scores_tuples, key=takeThird, reverse=True)

print('Find Target Disease Associations')
target_rank = np.zeros(len(sorted_tuples))

for i in range(len(sorted_tuples)):
    u,v,d = sorted_tuples[i]
    
    if (u,v) in target_edges:
        target_rank[i] = 1

    if i%10000000 == 0:
        print(i)


print('cummulative sum')
x = []
i = 1

while i < len(all_edges):
    i = i*10
    x.append(i)

y = [sum(target_rank[0:i]) for i in x]
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)
x_str = ["{}".format(fmt(i)) for i in x]

print('draw a plot')
plt.figure(figsize=(10,9))
plt.bar(x_str,y,color='#1f77b4', width=1.0, linewidth=0)
plt.title('Ranked Positive Edges')
plt.xlabel('# of Top Ranked Edges')
plt.ylabel('# of Positive Edges')
#plt.xticks(ticks=list(range(len(x))),labels=x)
plt.grid(b=True,which='major',axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('ranked_associations.png')
plt.close()

print(len(target_edges))
print(sum(target_rank))
