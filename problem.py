# noqa: D100
import numpy as np
import pandas as pd
import time as time
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

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
            label = "expression" + dico["gene"]
            scores = dico["scores"][0]
            if(len(scores) > 0):
                genes_dict[label] = scores
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


def get_all_survival_data(samples):
    hub = 'https://tcga.xenahubs.net'
    dataset = "survival/BRCA_survival.txt"
    samples = xena.dataset_samples(hub, dataset, None)
    # OS is 1 if the patient has died or 0 if he was censored (lost...)
    fields = ['DFI', 'DFI.time', 'DSS', 'DSS.time', 'OS', 'OS.time', 'PFI', 'PFI.time', 'sampleID']
    #fields = ['OS', "OS.time", "sampleID"]
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
# There are normal (e.g. not tumoral) samples in TCGA
# which should probably be removed. _sample_type will
# identify normals. _study will identify TCGA vs. GTEX vs. TARGET.
# Additionnaly, other co-variates, such as age, could be added


def get_phenotype_data(hub, samples):
    """Return the sample type (normal vs. tumoral) of the tumor."""
    dataset = "TCGA-BRCA.GDC_phenotype.tsv"
    fields = ['sample_type.samples', "sampleID"]
    values = get_fields_and_codes(hub, dataset, samples, fields)
    phenotype_dict = dict(zip(fields, values))  # index by phenotype
    phenotype_dict["sample_type"] = phenotype_dict.pop('sample_type.samples')
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

    # To remove duplicates and/or add supplementary phenotypical covariates
    if df_phenotype is not None:
        df = remove_duplicates(df, df_phenotype)
    df = df.drop('sampleID', axis=1)  # remove identifier for analysis
    return df


def download_data(database_filename):
    """Download and preprocess the expression and survival data."""
    # Transcriptomic data
    hub = "https://gdc.xenahubs.net"
    dataset = "TCGA-BRCA.htseq_fpkm-uq.tsv"
    samples = xena.dataset_samples(hub, dataset, None)
    df_expression = get_expression_data(hub, dataset, samples)

    # Survival data
    df_survival = get_survival_data(hub, dataset, samples)
    
    # Phenotype data
    # df_phenotype = get_phenotype_data(hub, samples)

    df_all = merge_data(df_expression, df_survival)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    database_path = os.path.join(data_dir, database_filename)
    filtered_df = filter_outliers(df_all)
    filtered_df.to_csv(database_path, index=False)
    return filtered_df
    

def load_or_download(database_filename):
    """Return the database (either by loading it, or by download it)."""
    data_dir = 'data'
    database_path = os.path.join(data_dir, database_filename)
    try:
        df = pd.read_csv(database_filename)
    except FileNotFoundError:
        df = download_data(database_filename)
    return df

def split_train_test(df):
    
    index_train, index_test = train_test_split( range(len(df)), test_size = 0.2, random_state=10)
    data_train = df.loc[index_train].reset_index( drop = True )
    data_test  = df.loc[index_test].reset_index( drop = True )


def filter_outliers(df):
    """The events occuring after 4500 are too rare and must be filtered, otherwise the Integrated Brier Score makes no sense."""
    long_times = df[df['time'] > 4500].index.to_numpy()
    return df.drop(long_times).reset_index()


def get_train_data():
    """Return the training data."""
    return

def get_test_data():
    """Return the test data."""
    return


problem_title = 'Breast cancer survival prediction'

score_types = [
    ConcordanceIndex(name='concordance_index'),
    IntegratedBrierScore(name='integrated_brier_score')
]


def get_cv(X, y):
    test = os.getenv('RAMP_TEST_MODE', 0)
    n_splits = 8
    if test:
        n_splits = 2
    spliter = GroupShuffleSplit(n_splits=n_splits, test_size=.2,
                            random_state=42)
    non_censored = X['death']
    splits = spliter.split(X, y, non_censored)

    # take only 500 samples per test subject for speed
    def limit_test_size(splits):
        rng = np.random.RandomState(42)
        for train, test in splits:
            yield (train, test[rng.permutation(len(test))[:500]])
    return limit_test_size(splits)


def _read_data(path, dir_name):
    DATA_HOME = path
    X_df = pd.read_csv(os.path.join(DATA_HOME,
                                    DATA_PATH,
                                    dir_name, 'X.csv.gz'))
    X_df.iloc[:, :-1] *= 1e12  # scale data to avoid tiny numbers

    # add a new column lead_field where you will insert a path to the
    # lead_field for each subject
    lead_field_files = os.path.join(DATA_HOME, DATA_PATH, '*L.npz')
    lead_field_files = sorted(glob.glob(lead_field_files))

    lead_subject = {}
    # add a row with the path to the correct LeadField to each subject
    for key in np.unique(X_df['subject']):
        path_L = [s for s in lead_field_files if key + '_L' in s][0]
        lead_subject[key] = path_L
    X_df['L_path'] = X_df.apply(lambda row: lead_subject[row.subject], axis=1)

    y = sparse.load_npz(
        os.path.join(DATA_HOME, DATA_PATH, dir_name, 'target.npz')).toarray()

        return X_df, y


def get_train_data(path="."):
    return _read_data(path, 'train')


def get_test_data(path="."):
    return _read_data(path, 'test')



#if __name__ == '__main__':
#    database_filename = 'test_database_TCGA.csv'
#   df = load_or_download(database_filename)
    
