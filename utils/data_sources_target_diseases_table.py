import csv

fout = open('data_sources_target_diseases_table.txt','w')

with open('v03/full.txt') as csvfile:
    file_reader = csv.reader(csvfile, delimiter='\t')
    
    print('Uniprot')
    line = next(file_reader)
    count = 0
    
    while line[0].find('CHEMBL') != 0:
        row = line[2].split('_')
        
        if (row[0] == 'GENE' and row[1] == 'DISEASE') or (row[0] == 'DISEASE' and row[1] == 'GENE'):
            count+=1
            
        line = next(file_reader)
        
        
    fout.write('Number of targets disease associations from UniProt is ' + str(count) + '\n')
    
    print('chembl')
    count = 0
    
    while line[0].find('CHEMBL') == 0:
        row = line[2].split('_')
        
        if (row[0] == 'GENE' and row[1] == 'DISEASE') or (row[0] == 'DISEASE' and row[1] == 'GENE'):
            count+=1
            
        line = next(file_reader)

    fout.write('Number of targets disease associations from ChEMBL is ' + str(count) + '\n')
    
    print('intact')
    count = 0
    
    while line[2].find('_intact_') > 0:
        row = line[2].split('_')
        
        if (row[0] == 'GENE' and row[1] == 'DISEASE') or (row[0] == 'DISEASE' and row[1] == 'GENE'):
            count+=1
            
        line = next(file_reader)

    fout.write('Number of targets disease associations from IntAct is ' + str(count) + '\n')
    
    print('open targets')
    count = 0
    
    while line[2].find('_nlp_') < 0:
        row = line[2].split('_')
        
        if (row[0] == 'GENE' and row[1] == 'DISEASE') or (row[0] == 'DISEASE' and row[1] == 'GENE'):
            count+=1
            
        line = next(file_reader)

    fout.write('Number of targets disease associations from Open Targets is ' + str(count) + '\n')
    
    print('LINK')
    count = 0
    
    row = line[2].split('_')
        
    if (row[0] == 'GENE' and row[1] == 'DISEASE') or (row[0] == 'DISEASE' and row[1] == 'GENE'):
        count+=1
    
    for line in file_reader:
        row = line[2].split('_')
        
        if (row[0] == 'GENE' and row[1] == 'DISEASE') or (row[0] == 'DISEASE' and row[1] == 'GENE'):
            count+=1

    fout.write('Number of targets disease associations from LINK is ' + str(count) + '\n')
    
    
fout.close()


