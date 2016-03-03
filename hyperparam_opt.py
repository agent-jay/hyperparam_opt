import numpy as np
import pandas as pd
from collections import defaultdict,OrderedDict
from scipy.stats import randint as sp_randint
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,log_loss,make_scorer
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,accuracy_score,matthews_corrcoef, make_scorer 
from sklearn.grid_search import ParameterGrid
from itertools import product
import logger
import pickle
import time
import json
from operator import itemgetter

msg=''

def param_get(n_iter, mode='rand'):
    '''Generates samples from the distribution'''
    if mode=='rand':
        print "Randomized search"
        for i in range(n_iter):
            params = {
            "n_estimators":sp_randint.rvs(10,200),
            "max_depth": random.choice([3, None]),
            "max_features": sp_randint.rvs(1, 20),
            "min_samples_split": sp_randint.rvs(1, 11),
            "min_samples_leaf": sp_randint.rvs(25, 100),
            "bootstrap": random.choice([True, False]),
            "criterion": random.choice(["gini", "entropy"]),
            "n_jobs":-1
            }
            yield params
    elif mode=='grid':
        print "Grid Search"
        param_list = {
        "n_estimators":range(1,501,10),
        "max_depth": [None],
        "max_features": ['auto'],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_leaf_nodes":[None],
        "bootstrap": [True],
        "criterion": ["entropy","gini"],
        "class_weight":["balanced"],
        "n_jobs":[-1]
        }
        for params in ParameterGrid(param_list):
            yield params

def main(log):
    results_dict= defaultdict(list) #store results that are converted to dataframe
    X_test=np.load('../cooked/NewSIFT_THD_Augmented_Train_X.npy')
    y_test=np.load('../cooked/NewSIFT_THD_Augmented_Train_y.npy')
    X_train=np.load('../cooked/NewSIFT_THD_Test_X.npy')
    y_train=np.load('../cooked/NewSIFT_THD_Test_y.npy')
    print "Data Loaded"
    
    iterate=10 #Set number of iterations
    print "Iterations to go through", iterate
    param_iter= param_get(n_iter= iterate, mode='grid') # is an iterator
    for i,params in enumerate(param_iter): 
        print "Iteration number:",i
        clf= RandomForestClassifier()
        clf.set_params(**params)
        #Fitting model
        start_time_fit  = time.time()
        clf.fit(X_train,y_train)
        end_time_fit = time.time()
        
        y_train_prob= clf.predict_proba(X_train)[:,1]
        auc_train= roc_auc_score(y_train, y_train_prob)
        #Test data
        start_time_pred  = time.time()
        y_test_prob= clf.predict_proba(X_test)[:,1]
        end_time_pred = time.time()
        
        #precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        auc_test= roc_auc_score(y_test, y_test_prob)
        
        #print "precision = ",precision
        #print "f1_score = ",f1_score
        #print "recall = ",recall 
        fit_time=end_time_fit - start_time_fit
        pred_time=end_time_pred - start_time_pred
        
        params['fitting time']=fit_time
        params['prediction time']=pred_time
        
        #Accumulating results in dataframe and log
        
        for key,value in clf.get_params(deep=True).items():
            print key,value
            results_dict[key].append(value)
         
        results_dict['fit_time'].append(fit_time)
        results_dict['pred_time'].append(pred_time)
        results_dict['auc_train'].append(auc_train)
        results_dict['auc_test'].append(auc_test)
        print "Score:", auc_test 
        log.info(str(params))
        
    results_df= pd.DataFrame(results_dict) 
    results_df.to_csv('./results/run_'+str(int(time.time()))+'.csv')
    print "Results saved to csv"

if __name__=='__main__':
    print "logger started"
    log = logger.create(filename='log_hyp.log',log_name="hyper_opti")
    log.info("Starting logger..")
    main(log=log) 
