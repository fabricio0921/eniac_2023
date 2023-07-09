import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SVMSMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids
from sklearn.metrics import accuracy_score, f1_score
import decodIRT_OtML as dIRT_OtML
import decodIRT_MLtIRT as dIRT_MLtIRT
import sys


def get_classication(resp,y):
    tmp = []
    for c,i in enumerate(y):
        if resp[c] == 1:
            tmp.append(i)
        elif i == 1:
            tmp.append(0)
        elif i == 0:
            tmp.append(1)
    return tmp

list_Datasets = ['ilpd','climate-model-simulation-crashes','credit-approval']

list_seeds = [10,20,30]
list_seeds_mtd = [10,20,30]

#list_over = ['adasyn','smote']
list_over = ['adasyn','smote','smoten','svmsmote','sem_nada']

list_clfs = ['GaussianNB','KNeighborsClassifier(8)','DecisionTreeClassifier()','RandomForestClassifier','SVM','MLPClassifier']
list_clfs_index = [0,5,6,9,10,11]

main_path = os.getcwd()+'/'

dict_scores = {}

for dataset in list_Datasets:
  dict_scores[dataset] = {}
  for seed_value in list_seeds:
    dict_scores[dataset]['seed_'+str(seed_value)] = {}
    for mtd_over in list_over:
      dict_scores[dataset]['seed_'+str(seed_value)][mtd_over] = {}
      for seed_mtd in list_seeds:
        #print('Calculando scores para {} seed {} do metodo {} do seed {}'.format(dataset,seed_value,mtd_over,seed_mtd))
        if mtd_over == 'sem_nada':
            path_result = dataset
        else:
            path_result = dataset+'_over_'+mtd_over+'_'+str(seed_mtd)

        df = pd.read_csv('seed_'+str(seed_value)+'/'+path_result+'/'+'acc_f1_score.csv', index_col=0)
        df_dict = df.transpose().to_dict()
        dict_scores[dataset]['seed_'+str(seed_value)][mtd_over]['seed_mtd_'+str(seed_mtd)] = df_dict

print('normal')
print(dict_scores['credit-approval']['seed_10']['sem_nada'])

dict_media_metd = {}

for dataset in list_Datasets:
  dict_media_metd[dataset] = {}
  for seed_value in list_seeds:
    dict_media_metd[dataset]['seed_'+str(seed_value)] = {}
    for mtd_over in list_over:
        dict_media_metd[dataset]['seed_'+str(seed_value)][mtd_over] = {}
        for clf in list_clfs:
            list_acc = [dict_scores[dataset]['seed_'+str(seed_value)][mtd_over][i][clf]['acc_score'] for i in dict_scores[dataset]['seed_'+str(seed_value)][mtd_over]]
            list_f1 = [dict_scores[dataset]['seed_'+str(seed_value)][mtd_over][i][clf]['f1_score'] for i in dict_scores[dataset]['seed_'+str(seed_value)][mtd_over]]

            tmp = {'acc_score':{'media':np.mean(list_acc), 'mediana':np.median(list_acc), 'desvio':np.std(list_acc)},
                'f1_score':{'media':np.mean(list_f1), 'mediana':np.median(list_f1), 'desvio':np.std(list_f1)}}
            dict_media_metd[dataset]['seed_'+str(seed_value)][mtd_over][clf] = tmp

print('teste media')
print(dict_media_metd['credit-approval']['seed_10']['sem_nada'])

dict_media_seed = {}

for dataset in list_Datasets:
    dict_media_seed[dataset] = {}
    for mtd_over in list_over:
        dict_media_seed[dataset][mtd_over] = {}
        for clf in list_clfs:
            list_acc = [dict_media_metd[dataset]['seed_'+str(i)][mtd_over][clf]['acc_score']['media'] for i in list_seeds]
            list_f1 = [dict_media_metd[dataset]['seed_'+str(i)][mtd_over][clf]['f1_score']['media'] for i in list_seeds]
            
            tmp = {'acc_score':{'media':round(np.mean(list_acc), 3), 'mediana':np.median(list_acc), 'desvio':np.std(list_acc)},
                'f1_score':{'media':round(np.mean(list_f1),3), 'mediana':np.median(list_f1), 'desvio':np.std(list_f1)}}
            dict_media_seed[dataset][mtd_over][clf] = tmp





print('####################################################################')
print('######################credit-approval x GaussianNB #########################')

print('####################################################################')


print('teste media seed adasyn')
print(dict_media_seed['credit-approval']['adasyn']['GaussianNB']['f1_score']['media'])

print('teste media seed smote')
print(dict_media_seed['credit-approval']['smote']['GaussianNB']['f1_score']['media'])

print('teste media seed')
print(dict_media_seed['credit-approval']['smoten']['GaussianNB']['f1_score']['media'])

print('teste media seed svmsmote')

print(dict_media_seed['credit-approval']['svmsmote']['GaussianNB']['f1_score']['media'])

print('####################################################################')
print('######################credit-approval x KNeighborsClassifier(8) #########################')

print('####################################################################')
print('teste media seed adasyn')
print(dict_media_seed['credit-approval']['adasyn']['KNeighborsClassifier(8)']['f1_score']['media'])

print('teste media seed smote')
print(dict_media_seed['credit-approval']['smote']['KNeighborsClassifier(8)']['f1_score']['media'])

print('teste media seed')
print(dict_media_seed['credit-approval']['smoten']['KNeighborsClassifier(8)']['f1_score']['media'])

print('teste media seed svmsmote')

print(dict_media_seed['credit-approval']['svmsmote']['KNeighborsClassifier(8)']['f1_score']['media'])

print('####################################################################')
print('######################credit-approval x DecisionTreeClassifier() #########################')

print('####################################################################')
print('teste media seed adasyn')
print(dict_media_seed['credit-approval']['adasyn']['DecisionTreeClassifier()']['f1_score']['media'])

print('teste media seed smote')
print(dict_media_seed['credit-approval']['smote']['DecisionTreeClassifier()']['f1_score']['media'])

print('teste media seed')
print(dict_media_seed['credit-approval']['smoten']['DecisionTreeClassifier()']['f1_score']['media'])

print('teste media seed svmsmote')

print(dict_media_seed['credit-approval']['svmsmote']['DecisionTreeClassifier()']['f1_score']['media'])

print('####################################################################')
print('######################credit-approval x RandomForestClassifier #########################')

print('####################################################################')
print('teste media seed adasyn')
print(dict_media_seed['credit-approval']['adasyn']['RandomForestClassifier']['f1_score']['media'])

print('teste media seed smote')
print(dict_media_seed['credit-approval']['smote']['RandomForestClassifier']['f1_score']['media'])

print('teste media seed')
print(dict_media_seed['credit-approval']['smoten']['RandomForestClassifier']['f1_score']['media'])

print('teste media seed svmsmote')

print(dict_media_seed['credit-approval']['svmsmote']['RandomForestClassifier']['f1_score']['media'])

print('####################################################################')
print('######################credit-approval x SVM #########################')

print('####################################################################')
print('teste media seed adasyn')
print(dict_media_seed['credit-approval']['adasyn']['SVM']['f1_score']['media'])

print('teste media seed smote')
print(dict_media_seed['credit-approval']['smote']['SVM']['f1_score']['media'])

print('teste media seed')
print(dict_media_seed['credit-approval']['smoten']['SVM']['f1_score']['media'])

print('teste media seed svmsmote')

print(dict_media_seed['credit-approval']['svmsmote']['SVM']['f1_score']['media'])

print('####################################################################')
print('######################credit-approval x MLPClassifier #########################')

print('####################################################################')
print('teste media seed adasyn')
print(dict_media_seed['credit-approval']['adasyn']['MLPClassifier']['f1_score']['media'])

print('teste media seed smote')
print(dict_media_seed['credit-approval']['smote']['MLPClassifier']['f1_score']['media'])

print('teste media seed')
print(dict_media_seed['credit-approval']['smoten']['MLPClassifier']['f1_score']['media'])

print('teste media seed svmsmote')

print(dict_media_seed['credit-approval']['svmsmote']['MLPClassifier']['f1_score']['media'])




