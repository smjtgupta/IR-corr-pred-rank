import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from scipy import stats

train_data = "result_system_train.csv" # change it accordingly
web2012_test_data = "result_system_wt2012.csv"
web2013_test_data = "result_system_wt2013.csv"
web2014_test_data = "result_system_wt2014.csv"


system_train = pd.read_csv(train_data)
web2012_test = pd.read_csv(web2012_test_data)
web2013_test = pd.read_csv(web2013_test_data)
web2014_test = pd.read_csv(web2014_test_data)


# Select the set of metrics you would like to try
s = ['Rprecall', 'bprefall','recip_rankall','errall' ,    'mapall',#'map_cut_5all','map_cut_10all','map_cut_15all','map_cut_20all','map_cut_30all','map_cut_100all','map_cut_200all','map_cut_500all',
   #  'gm_mapall',
'ndcgall',#'ndcg_cut_5all','ndcg_cut_10all','ndcg_cut_20all','ndcg_cut_30all','ndcg_cut_100all','ndcg_cut_200all','ndcg_cut_500all',
    #'P_5all',
    'P_10all',#'P_20all',#'P_30all', 'P_100all','P_200all','P_500all','P_1000all',
    #'recall_5all',
    'recall_100all',
    #'recall_20all','recall_100all','recall_200all','recall_500all','recall_1000all','gm_mapall',
    'rdp_5all','rdp_8all','rdp_95all'
    ]

system_train = system_train[s]
web2012_test = web2012_test[s]
web2013_test = web2013_test[s]
web2014_test = web2014_test[s]

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
    y_pred_test = model.predict(x_test)
    
    
    MSE = np.mean((y_pred_test - y_test)**2)  # Mean Square Error
    R_sq_test = sklearn.metrics.r2_score(y_test, y_pred_test)
    tau,p_value = stats.kendalltau(y_test, y_pred_test)
    
    if printing == True:
        print("%d|%s|%s|%s|%.3lf|%.3lf|%.3lf" % (test.shape[0],str(m),str(n),str(coefficients), MSE**0.5, tau[0],R_sq_test))
    
    return (tau,MSE**0.5,R_sq_test, coefficients,p_value )

def find_best(predict):
    max_r = -1000000
    max_tau = -1
    for j in range(0,len(s)):
        if s[j] == predict:
            continue
 
       
        feature = [s[j]]
        curr_result = IREM(system_train,web2012_test,[predict],feature, False)

        if max_tau < curr_result[0]:
            #max_r = curr_result[2]
            max_tau = curr_result[0]
            selected_feature = feature
            coefficient = curr_result[3][0]
            tau = curr_result[0]
            rmse = curr_result[1][0]
            r = curr_result[2]
            tau_p = curr_result[4]
            
    tau_13 = IREM(system_train,web2013_test,[predict],selected_feature)
    tau_14 = IREM(system_train,web2014_test,[predict],selected_feature)

    print("%s|%s|%.3f|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf" %(predict,str(selected_feature), 
                    tau,tau_p,r,tau_13[0], tau_13[4], tau_13[2],  tau_14[0], tau_14[4], tau_14[2] ))


def find_best_2(predict):
    max_tau = -1
    for j in range(0,len(s)):
        if s[j] == predict:
            continue

        for k in range(j+1,len(s)):
            if s[k] == predict:
                continue
            feature = [s[j],s[k]]
            curr_result = IREM(system_train,web2012_test,[predict],feature)
            if max_tau < curr_result[0]:
                max_tau = curr_result[0]
                selected_feature = feature
                tau = curr_result[0]
                r = curr_result[2]
                tau_p = curr_result[4]

    tau_13 = IREM(system_train,web2013_test,[predict],selected_feature)
    tau_14 = IREM(system_train,web2014_test,[predict],selected_feature)

    print("%s|%s|%.3f|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf" %(predict,str(selected_feature), 
                    tau,tau_p,r,tau_13[0], tau_13[4], tau_13[2],  tau_14[0], tau_14[4], tau_14[2] ))

def find_best_3(predict):
    max_tau = -1
    max_r = -1000000
    for j in range(0,len(s)):
        if s[j] == predict:
            continue
    
        for k in range(j+1,len(s)):
            if s[k] == predict:
                continue
      
            for m in range(k+1,len(s)):
                if s[m] == predict:
                    continue
    
                feature = [s[j],s[k],s[m]]
                curr_result = IREM(system_train,web2012_test,[predict],feature)
                if max_tau < curr_result[0]:
                    max_r = curr_result[2]
                    max_tau = curr_result[0]
                    selected_feature = feature
                    tau = curr_result[0]
                    r = curr_result[2]
                    tau_p = curr_result[4]
    tau_13 = IREM(system_train,web2013_test,[predict],selected_feature)
    tau_14 = IREM(system_train,web2014_test,[predict],selected_feature)

    print("%s|%s|%.3f|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf|%.3lf" %(predict,str(selected_feature), 
                    tau,tau_p,r,tau_13[0], tau_13[4], tau_13[2],  tau_14[0], tau_14[4], tau_14[2] ))

 
def exp_with_power_set_1():
    for i in range(0,len(s)):
        find_best(s[i])

def exp_with_power_set_2():
    for i in range(0,len(s)):
        find_best_2(s[i])

def exp_with_power_set_3():
    for i in range(0,len(s)):
        find_best_3(s[i])
        

exp_with_power_set_1()
exp_with_power_set_2()
exp_with_power_set_3()
