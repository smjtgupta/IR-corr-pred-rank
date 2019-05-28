import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import scipy.stats as stats

from itertools import *

train_data = "result_system_train_low_to_high.csv" # change it accordingly
web2012_test_data = "result_web2012_low_to_high.csv"
web2013_test_data = "result_web2013_low_to_high.csv"
web2014_test_data = "result_web2014_low_to_high.csv"


system_train = pd.read_csv(train_data)
web2012_test = pd.read_csv(web2012_test_data)
web2013_test = pd.read_csv(web2013_test_data)
web2014_test = pd.read_csv(web2014_test_data)




s = ['1000_P_1000all','1000_bprefall','1000_errall', '1000_infAPall','1000_mapall','1000_ndcgall',
    '100_P_1000all','100_bprefall','100_errall', '100_infAPall','100_mapall','100_ndcgall',
    '10_P_1000all','10_bprefall','10_errall', '10_infAPall','10_mapall','10_ndcgall',
    '20_P_1000all','20_bprefall','20_errall', '20_infAPall','20_mapall','20_ndcgall',
    ]

low_metric_10 = ['10_P_1000all','10_bprefall','10_errall', '10_infAPall','10_mapall','10_ndcgall','10_rbp_95all']
low_metric_20 = ['20_P_1000all','20_bprefall','20_errall', '20_infAPall','20_mapall','20_ndcgall','20_rbp_95all']
low_metric_30 = ['30_P_1000all','30_bprefall','30_errall', '30_infAPall','30_mapall','30_ndcgall','30_rbp_95all']
low_metric_40 = ['40_P_1000all','40_bprefall','40_errall', '40_infAPall','40_mapall','40_ndcgall','40_rbp_95all']
low_metric_50 = ['50_P_1000all','50_bprefall','50_errall', '50_infAPall','50_mapall','50_ndcgall','50_rbp_95all']

high_cost_metric = ['1000_P_1000all','1000_mapall','1000_ndcgall', '1000_rbp_95all',
                    '100_P_1000all','100_mapall','100_ndcgall','100_rbp_95all']    

system_train = system_train.dropna(how = 'all')
web2012_test = web2012_test.dropna(how = 'all')
web2013_test = web2013_test.dropna(how = 'all')
web2014_test = web2014_test.dropna(how = 'all')

def IREM(train,test, m = ['map'], n = ['Rprec', 'bpref', 'recip_rank', 'P_10','P_100'], printing =False):
    y_train, x_train, y_test, x_test = train[m], train[n], test[m], test[n]
    
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    
    coefficients = model.coef_
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    MSE = np.mean((y_pred_test - y_test)**2)  # Mean Square Error
    R_sq_test = sklearn.metrics.r2_score(y_test, y_pred_test)
    tau = stats.kendalltau(y_test, y_pred_test)
    
    if printing == True:
        print("%d|%s|%s|%s|%.3lf|%.3lf|%.3lf" % (test.shape[0],str(m),str(n),str(coefficients), MSE**0.5, tau[0],R_sq_test))
    return (tau[0],MSE**0.5, R_sq_test,coefficients )

def find_power_set(metric_to_be_predicted, features):
    power_set = list(chain.from_iterable(combinations(features, r) for r in range(1, len(features) )))
    max_tau = -1;
    for sf in power_set:
        features= []
        for i in range(0,len(sf)):
            features.append(sf[i])
        
        curr_result = IREM(system_train,web2012_test,[metric_to_be_predicted],features, False)
        if max_tau < curr_result[0]:
            max_tau = curr_result[0]
            selected_feature = features
            coefficient = curr_result[3][0]
            rmse = curr_result[1][0]
            r = curr_result[2]
    tau_12 = IREM(system_train,web2012_test,[metric_to_be_predicted],selected_feature)
    tau_13 = IREM(system_train,web2013_test,[metric_to_be_predicted],selected_feature)
    tau_14 = IREM(system_train,web2014_test,[metric_to_be_predicted],selected_feature)
    print("%s|%s|%s|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf" %(metric_to_be_predicted,str(selected_feature),str(coefficient),
                                        max_tau,rmse,tau_13[0],tau_13[1][0],tau_14[0],tau_14[1][0],
                                        r,tau_13[2],tau_14[2] ))

   
for i in range(0,len(high_cost_metric)):
    find_power_set( high_cost_metric[i],low_metric_10)
    find_power_set( high_cost_metric[i],low_metric_20)
    find_power_set( high_cost_metric[i],low_metric_30)
    find_power_set( high_cost_metric[i],low_metric_40)
    find_power_set( high_cost_metric[i],low_metric_50)
    