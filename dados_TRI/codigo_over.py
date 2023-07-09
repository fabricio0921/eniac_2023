import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SVMSMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids
import decodIRT_OtML as dIRT_OtML
import decodIRT_MLtIRT as dIRT_MLtIRT

list_Datasets = ['ilpd']#,'climate-model-simulation-crashes','credit-approval']

list_seeds = [30]
list_seeds_mtd = [10,20,30]

#list_over = ['adasyn','smote']
list_over = ['adasyn','smote','smoten','svmsmote']

list_under = []

main_path = os.getcwd()+'/'

for dataset in list_Datasets:
  for seed_value in list_seeds:

    out_path = 'seed_'+str(seed_value)
    dIRT_OtML.main(arg_data=dataset,arg_dataset=None,arg_dataTest=None,
               arg_saveData=True,arg_seed=seed_value,arg_output=out_path)
    
    data_test_path = out_path+'/'+dataset+'/'+dataset+'_test.csv'

    df = pd.read_csv(out_path+'/'+dataset+'/'+dataset+'_train.csv', index_col=False)
    X = df.drop(['class'], axis=1)
    y = df['class']
    y=y.astype('int')

    for seed_mtd in list_seeds_mtd:

      dict_over = {'adasyn':ADASYN(random_state=seed_mtd),'smote':SMOTE(random_state=seed_mtd),'smoten':SMOTEN(random_state=seed_mtd),'svmsmote':SMOTEN(random_state=seed_mtd)}
      dict_under = {'nearmiss':NearMiss(),'ClusterCentroids':ClusterCentroids(random_state=seed_mtd)}

      for over_mtd in list_over:
        #ros = ADASYN(random_state=10) # String
        X_over, y_over = dict_over[over_mtd].fit_resample(X, y)

        X_over['class'] = list(y_over)
        save_csv = main_path+dataset+'_over_'+over_mtd+'_'+str(seed_mtd)+'.csv'
        X_over.to_csv(save_csv,index=False)

        dIRT_OtML.main(arg_data=None,arg_dataset=save_csv,arg_dataTest=data_test_path,
                arg_saveData=True,arg_seed=seed_value,arg_output=out_path)

      for under_mtd in list_under:
        #ros = ADASYN(random_state=10) # String
        X_under, y_under = dict_under[under_mtd].fit_resample(X, y)

        X_under['class'] = list(y_under)
        save_csv = main_path+dataset+'_under_'+under_mtd+'_'+str(seed_mtd)+'.csv'
        X_under.to_csv(save_csv,index=False)

        dIRT_OtML.main(arg_data=None,arg_dataset=save_csv,arg_dataTest=data_test_path,
                arg_saveData=True,arg_seed=seed_value,arg_output=out_path)
    
    dIRT_MLtIRT.main(arg_dir = out_path)