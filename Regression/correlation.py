import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('ggplot')
from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

system_based_data = "result_system_all.csv" # change it based on your setup
query_based_data = "result_query_all.csv" # change it based on your setup

def plot_heat_map(query_corr, subtitle, filename,labels):
    
    plt.figure(figsize=(4,4));
    cax = plt.imshow(query_corr, cmap='RdYlGn', interpolation='none', aspect='auto');#RdYlGn
    plt.colorbar();
    
    plt.xticks(range(len(query_corr)), labels,fontsize=8, rotation='vertical');
    plt.yticks(range(len(query_corr)), labels,fontsize=8);
    plt.savefig(filename, bbox_inches='tight');
    plt.show();

system = pd.read_csv(system_based_data)
query = pd.read_csv(query_based_data)

# select the metic you would like to find the correlation
q = ['infAP',
    'Rprec', 'bpref','recip_rank','err', 
     'map_cut_5',
    'map_cut_10', 'map_cut_20','map_cut_30',
    'map_cut_100','map_cut_200','map_cut_500',
    'map','ndcg_cut_5',
    'ndcg_cut_10','ndcg_cut_20','ndcg_cut_30',
    'ndcg_cut_100','ndcg_cut_200','ndcg_cut_500','ndcg',
    'P_5',  'P_10','P_20','P_30', 'P_100','P_200','P_500','P_1000',
    'recall_5', 'recall_10','recall_20', 'recall_100','recall_200','recall_500', 'recall_1000',
    'rdp_5','rdp_8','rdp_95']
    

query = query[q]

query = query.dropna(how = 'all')

query_corr_pearson = query.corr(method='pearson')
type(query_corr_pearson)

q_labels = ['infAP',
    'R-Prec', 'bpref','RR', 'ERR',
     'MAP@5',
    'AP@10','AP@20','MAP@30',
    'AP@100','MAP@200','MAP@500',
    'AP@1000',
    'AP',
     'NDCG@5',
    'nDCG@10','nDCG@20','NDCG@30',
    'nDCG@100','NDCG@200','NDCG@500',
    'nDCG',
    'nDCG@1000',
    'P@5',
    'P@10','P@20','P@30',
    'P@100','P@200','P@500',
    'P@1000',
    'R@5',
    'R@10','R@20',
    'R@100','R@200','R@500',
    'R@100',
    'RBP-0.5','RBP-0.8','RBP-0.95']

plot_heat_map(query_corr_pearson,'IR System EM (query-wise) Pearson Correlations Heat Map','heatmap_pearson_query_wise.png',q_labels)

