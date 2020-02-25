# -*- coding: utf-8 -*-
import csv

fout = open('data_sources_table.txt','w')

with open('v03/full.txt') as csvfile:
    file_reader = csv.reader(csvfile, delimiter='\t')
    
    print('Uniprot')
    line = next(file_reader)
    count = 0
    edges_type = set()
    
    while line[0].find('CHEMBL') != 0:
        count+=1
        row = line[2].split('_')[0:2]
        edges_type.add(row[0]+'___'+row[1])
        line = next(file_reader)
        
        
    fout.write('Number of edges from UniProt is ' + str(count) + '\n')
    fout.write('Types of edges from UniProt:\n')
    fout.write(str(edges_type))
    fout.write('\n')
    
    print('chembl')
    count = 0
    edges_type = set()
    
    while line[0].find('CHEMBL') == 0:
        count+=1
        row = line[2].split('_')[0:2]
        edges_type.add(row[0]+'___'+row[1])
        line = next(file_reader)

    fout.write('Number of edges from ChEMBL is ' + str(count) + '\n')
    fout.write('Types of edges from ChEMBL:\n')
    fout.write(str(edges_type))
    fout.write('\n')
    
    print('intact')
    count = 0
    edges_type = set()
    
    while line[2].find('_intact_') > 0:
        count+=1
        row = line[2].split('_')[0:2]
        edges_type.add(row[0]+'___'+row[1])
        line = next(file_reader)

    fout.write('Number of edges from IntAct is ' + str(count) + '\n')
    fout.write('Types of edges from IntAct:\n')
    fout.write(str(edges_type))
    fout.write('\n')
    
    print('open targets')
    count = 0
    edges_type = set()
    
    while line[2].find('_nlp_') < 0:
        count+=1
        row = line[2].split('_')[0:2]
        edges_type.add(row[0]+'___'+row[1])
        line = next(file_reader)

    fout.write('Number of edges from Open Targets is ' + str(count) + '\n')
    fout.write('Types of edges from Open Targets:\n')
    fout.write(str(edges_type))
    fout.write('\n')
    
    print('LINK')
    count = 1
    edges_type = set()
    row = line[2].split('_')[0:2]
    edges_type.add(row[0]+'___'+row[1])
    
    for line in file_reader:
        count+=1
        row = line[2].split('_')[0:2]
        edges_type.add(row[0]+'___'+row[1])

    fout.write('Number of edges from LINK is ' + str(count) + '\n')
    fout.write('Types of edges from LINK:\n')
    fout.write(str(edges_type))
    fout.write('\n')
    
fout.close()
