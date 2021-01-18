# noqa: D100
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import time as time
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# for gene names converting
import mygene as mg

# for data fetching
import xenaPython as xena

data_dir = 'data'

def get_codes(host, dataset, fields, data):
    """Get codes for enumerations."""
    codes = xena.field_codes(host, dataset, fields)
    codes_idx = dict([(x['name'], x['code'].split('\t'))
                      for x in codes if x['code'] is not None])
    for i in range(len(fields)):
        if fields[i] in codes_idx:
            data[i] = [None if v == 'NaN' else codes_idx[fields[i]][int(v)]
                       for v in data[i]]
    return data


def get_fields(host, dataset, samples, fields):
    """Get field values."""
    fields = xena.dataset_fetch(host, dataset, samples, fields)
    return fields


def get_fields_and_codes(host, dataset, samples, fields):
    """Get fields and resolve codes."""
    return get_codes(host, dataset, fields,
                     get_fields(host, dataset, samples, fields))


def available_genes(hub, dataset, fields):
    """Return all the non-duplicated genes in the dataset."""
    genes = xena.dataset_field(hub, dataset)[:-1]

    for i in range(len(genes)):
        genes[i] = genes[i].split(".")[0]
    # removing the digits after the ENSG creates duplicates

    # Convert gene names into their symbol
    dico_genes = mg.get_client("gene")
    dico_genes.getgenes
    dico_genes = mg.get_client("gene").getgenes(genes, 'symbol')

    gene_symbols = [d.get('symbol') for d in dico_genes]
    gene_symbols = [x for x in gene_symbols if x is not None]

    # Eliminate the duplicates in the gene names
    distinct_genes = list(set(gene_symbols))
    # distinct_genes = [x for x in gene_symbols
    # if x not in genes_set and not genes_set.add(x)]
    # genes_test = list(genes_set)
    # assert genes_set == set(distinct_genes)
    print("We collect the data for %s genes."
          % str(len(distinct_genes)))
    return distinct_genes


def gather_expression_data(hub, dataset, samples, genes):
    """Collect the expression data from Xena Hub."""
    expression_data = np.array([])
    n_genes = len(genes)
    t_0 = time.time()
    # We collect the expression data 100 genes by 100 genes
    for i in range(int(n_genes/100) + 1):
        if i % 20 == 0:
            print('%i genes collected in %s s'
                  % (100*i, str(time.time() - t_0)))
        lower_bound = min(i*100, n_genes)
        upper_bound = min((i+1)*100, n_genes)
        genes_batch = genes[lower_bound: upper_bound]
        new_expression_batch = np.array(xena.dataset_gene_probe_avg(
                                  hub, dataset, samples, genes_batch))
        expression_data = np.append(expression_data, new_expression_batch)
    return expression_data.tolist()


def match_id_value(hub, dataset, samples, fields, expression_data):
    """Return a dict with the expression data associated to each gene."""
    values = get_fields_and_codes(hub, dataset, samples, fields)
    genes_dict = dict(zip(fields, values))
    for dico in expression_data:
        if dico["gene"] is not None:
            scores = dico["scores"][0]
            if(len(scores) > 0):
                genes_dict[dico["gene"]] = scores
    df_genes_expression = pd.DataFrame(data=genes_dict)
    return df_genes_expression


def get_survival_data(hub, dataset, samples):
    """Return a Dataframe containing sampleID and survival value."""
    dataset = "TCGA-BRCA.survival.tsv"

    # OS is 1 if the patient has died or 0 if he was censored (lost...)
    fields = ['OS', "OS.time", "sampleID"]
    values = get_fields_and_codes(hub, dataset, samples, fields)
    survival_dict = dict(zip(fields, values))  # index by phenotype
    survival_dict["time"] = survival_dict.pop("OS.time")
    survival_dict["death"] = survival_dict.pop("OS")
    df_survival = pd.DataFrame(data=survival_dict)
    return df_survival


def get_expression_data(hub, dataset, samples):
    """Return a Dataframe containing sampleID and expression data."""
    fields = ["sampleID"]
    genes = available_genes(hub, dataset, fields)
    expression_data = gather_expression_data(hub, dataset, samples, genes)
    df_genes_expression = match_id_value(hub, dataset,
                                         samples, fields, expression_data)
    return df_genes_expression

# WARNING : this code is not used for the moment
# It could allow to add other co-variates, such as age.

def get_phenotype_data(hub, samples):
    """Return phenotypic data of each sample."""
    dataset = "TCGA-BRCA.GDC_phenotype.tsv"
    values = get_fields_and_codes(hub, dataset, samples)
    phenotype_dict = dict(zip(fields, values))  # index by phenotype
    df_phenotype = pd.DataFrame(data=phenotype_dict)
    return df_phenotype


def remove_duplicates(df_total, df_phenotype):
    """Filter the duplicates (e.g. keep only the primary tumors)."""
    df_total = pd.merge(df_total, df_phenotype)
    df_total = df_total[df_total["sample_type"] == "Primary Tumor"]
    df_total = df_total.drop(['sample_type'], axis=1)
    return df_total


def merge_data(df_expression, df_survival, df_phenotype=None):
    """Merge the expression data with the survival data."""
    df = pd.merge(df_expression, df_survival)  # Merge according to sampleID
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[~pd.isna(df["time"])][~pd.isna(df["death"])]
    df = df.drop('sampleID', axis=1)  # remove identifier for analysis
    return df


def download_data():
    """Download and preprocess the expression and survival data."""
    # Transcriptomic data
    hub = "https://gdc.xenahubs.net"
    dataset = "TCGA-BRCA.htseq_fpkm-uq.tsv"
    samples = xena.dataset_samples(hub, dataset, None)
    df_expression = get_expression_data(hub, dataset, samples)

    # Survival data
    df_survival = get_survival_data(hub, dataset, samples)
    
    # Phenotype data (unused for now) : could be used to add covariates
    # df_phenotype = get_phenotype_data(hub, samples)

    df_all = merge_data(df_expression, df_survival)
    filtered_df = filter_outliers(df_all)
    record_train_test(filtered_df)
    return 
    

def check_previous_download():
    """Download the data if they didn't existed."""
    dirs = ['%s/train' % data_dir, '%s/test' % data_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            download_data()
        else:
            print('%s were already downloaded.' % directory)


def record_train_test(df):
    """Split and record the data."""
    # Split
    index_train, index_test = train_test_split(range(len(df)), test_size = 0.2, random_state=10)
    df_train = df.loc[index_train].reset_index(drop = True)
    df_test  = df.loc[index_test].reset_index(drop = True)
    
    # Create the architecture to record the data
    train_dir, test_dir = os.path.join(data_dir, 'train'), os.path.join(data_dir, 'test')
    for directory in [train_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Record the dataframes
    record(df_train, train_dir)
    record(df_test, test_dir)
    

def record(df, directory="."):
    """Split a dataframe into X and y and record it."""
    X_path = os.path.join(directory, 'X.csv')
    y_path = os.path.join(directory, 'y.csv')
    paths = [X_path, y_path]
    X = df
    y = pd.concat([X.pop('death'), X.pop('time')], ignore_index=True, axis=1)
    y.rename({0:'death', 1:'time'}, inplace=True, axis=1)
    for i, dataframe in enumerate([X, y]):
        dataframe.to_csv(paths[i], index = False)
    

def filter_outliers(df):
    """The events occuring after 4500 are too rare and must be filtered, otherwise the Integrated Brier Score makes no sense."""
    long_times = df[df['time'] > 4500].index.to_numpy()
    return df.drop(long_times).reset_index(drop=True)


if __name__ == '__main__':
    check_previous_download()
