import numpy as np
import pandas as pd
import csv as csv

#for gene names converting
import mygene as mg

#for finding duplicates
import collections

#for data fetching
import xenaPython as xena

def get_codes(host, dataset, fields, data):
    "get codes for enumerations"
    codes = xena.field_codes(host, dataset, fields)
    codes_idx = dict([(x['name'], x['code'].split('\t')) for x in codes if x['code'] is not None])
    for i in range(len(fields)):
        if fields[i] in codes_idx:
            data[i] = [None if v == 'NaN' else codes_idx[fields[i]][int(v)] for v in data[i]]
    return data


def get_fields(host, dataset, samples, fields):
    "get field values"
    data = xena.dataset_fetch(host, dataset, samples, fields)
    return data


def get_fields_and_codes(host, dataset, samples, fields):
    "get fields and resolve codes"
    return get_codes( host, dataset, fields, get_fields( host, dataset, samples, fields))


#### Gene expression data

dico_genes = mg.get_client("gene")
dico_genes.getgenes

hub = "https://gdc.xenahubs.net"
dataset = "TCGA-BRCA.htseq_fpkm-uq.tsv" #database url was found on the hub website
samples = xena.dataset_samples (hub, dataset, None)

### To download specific genes

##wave genes
#genes =["NCKAP1","CYFIP2", "NCKAP1L", "WASF2", "ABI3", "WASF3", "ABI1", "ABI2", "CYFIP1", "WASF1", "BRK1"] #enter the genes of interest
##Arp2-3 genes
#genes = ["ACTR2", "ACTR3", "ARPC1A", "ARPC2", "ARPC3", "ARPC4", "ARPC5", "ACTR3B", "ARPC1B", "ARPC5L"]
##Arpin
#genes = ["Arpin"]


## All genes available
genes = xena.dataset_field(hub, dataset)[:-1]
fields = ["sampleID"]

for i in range(len(genes)):
    genes[i] = genes[i].split(".")[0] #What is the meaning of the number after the point ?
    
#Convert gene names into their symbol
genes_1 = []
dico_genes = mg.get_client("gene").getgenes(genes,'symbol') #Enlever les chiffres après le point dans le ENSG : create duplicates
genes_1 = [d.get('symbol') for d in dico_genes]
genes_1 = [x for x in genes_1 if x is not None]

#Eliminate the duplicates in the gene names
genes_set = set()
genes_diff = [x for x in genes_1 if x not in genes_set and not genes_set.add(x)]
print('len(genes_diff )',len(genes_diff))
print('len(genes_1) ',len(genes_1))

######
expressions_0 = np.array([])

for i in range(1): # Maybe to modify TODO
    if i % 20 == 0:
        print(i)
    expressions_0 = np.append(expressions_0,np.array(xena.dataset_gene_probe_avg(hub, dataset, samples, genes_diff[min(i*100, len(genes_diff)) : min((i+1)*100, len(genes_diff))])))
    print(xena.dataset_gene_probe_avg(hub, dataset, samples, genes_diff[min(i*100, len(genes_diff)) : min((i+1)*100, len(genes_diff))]))
    #expressions_0.extend(xena.dataset_gene_probe_avg(hub, dataset, samples, genes_diff[min(i*100, len(genes_diff)) : min((i+1)*100, len(genes_diff))]))
#expressions_0 = xena.dataset_gene_probe_avg(hub, dataset, samples, genes)

expressions_1 = expressions_0.tolist()

with open('expressions_1_partial', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(expressions_1)    
    
values = get_fields_and_codes(hub, dataset, samples, fields) # list of lists

genes_dict_for_df = dict(zip(fields, values)) #dict where we add the gene expression and the sample ids

with open('dict_partial.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in genes_dict_for_df.items():
       writer.writerow([key, value])
       
for dico in expressions_1:
    if(dico["gene"] != None):
        label = "expression" + dico["gene"]
        scores = dico["scores"][0] #Problem with the filfulling of the scores of expression_0 : it needs probably the usual names of the genes (like Arpin) instead of the scientific names (like ENSG0000000.03) 
        if(len(scores) > 0):
            genes_dict_for_df[label] = scores
            
#### Survival data

dataset = "TCGA-BRCA.survival.tsv"
fields = ['_TIME_TO_EVENT', "_EVENT", "sampleID"] #event is 1 if the patient has died or 0 if he was censored (lost...)
#TODO
# As in pancan, there are normal samples in tcga which should probably be removed. _sample_type will
# identify normals. _study will identify tcga vs. gtex vs. target.
values = get_fields_and_codes(hub, dataset, samples, fields) # list of lists
survival_dict_for_df = dict(zip(fields, values)) # index by phenotype
survival_dict_for_df["survival"] = survival_dict_for_df.pop("_TIME_TO_EVENT") 
survival_dict_for_df["event"] = survival_dict_for_df.pop("_EVENT")

#### Phenotypes data

dataset = "TCGA-BRCA.GDC_phenotype.tsv"
fields = ['sample_type.samples', "sampleID"] 
values = get_fields_and_codes(hub, dataset, samples, fields) # list of lists
phenotype_dict_for_df = dict(zip(fields, values)) # index by phenotype
phenotype_dict_for_df["sample_type"] = phenotype_dict_for_df.pop('sample_type.samples')

"""I don't know if it's useful to remove the patients whose tumor is not a "Primary Tumor", like advised by Xena. The goal is to remove duplicates but I can't find duplicates"""

#### Merge data

df_genes = pd.DataFrame(data=genes_dict_for_df)
df_survival = pd.DataFrame(data=survival_dict_for_df)
#df_phenotype = pd.DataFrame(data=phenotype_dict_for_df)

df = pd.merge(df_genes, df_survival) # merge the survival and genes data according to sampleID
#df = pd.merge(df, df_phenotype) #merge also the phenotype data

#df = df.convert_objects(convert_numeric=True)
df = pd.to_numeric(df)

df = df[~pd.isna(df["survival"])][~pd.isna(df["event"])]
#df = df[df["sample_type"] == "Primary Tumor"] #keep only primary tumor to remove duplicate WARNING : check if name doesn't change if you use another dataset
#df = df.drop(['sampleID','sample_type'], axis=1) #remove identifier for analysis

df = df.drop('sampleID', axis=1) #remove identifier for analysis

# Record the data
df.to_csv("data_base_partial.csv")
