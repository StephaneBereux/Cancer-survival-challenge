# noqa: D100
import numpy as np
import pandas as pd
import time as time

# for gene names converting
import mygene as mg

# for data fetching
import xenaPython as xena


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
    survival_dict["survival"] = survival_dict.pop("OS.time")
    survival_dict["death"] = survival_dict.pop("OS")
    df_survival = pd.DataFrame(data=survival_dict)
    return df_survival


def get_expression_data(hub, dataset, samples):
    """Return a Dataframe containing sampleID and expression data."""
    fields = ["sampleID"]
    genes = available_genes(hub, dataset, fields)
    genes = genes[:90]
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
    df = df[~pd.isna(df["survival"])][~pd.isna(df["death"])]

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
    df_all.to_csv(database_filename, index=False)

    return df_all


def load_or_download(database_filename):
    """Return the database (either by loading it, or by download it)."""
    try:
        df = pd.read_csv(database_filename)
    except FileNotFoundError:
        df = download_data(database_filename)
    return df


if __name__ == '__main__':
    database_filename = 'database_TCGA.csv'
    load_or_download(database_filename)
