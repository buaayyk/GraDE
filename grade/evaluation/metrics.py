import numpy as np
from scipy.stats import spearmanr, kendalltau

def spearman_correlation(estimate_values, topK_subgraphs_count):
    topK_subgraphs_count = np.array(topK_subgraphs_count)
    estimate_values = np.array(estimate_values)
    correlation, pvalue = spearmanr(estimate_values, topK_subgraphs_count)
    return correlation, pvalue

def kendall_correlation(estimate_values, topK_subgraphs_count):
    estimate_values = np.array(estimate_values)
    topK_subgraphs_count = np.array(topK_subgraphs_count)
    correlation, pvalue = kendalltau(estimate_values, topK_subgraphs_count, nan_policy='omit')
    return correlation, pvalue
