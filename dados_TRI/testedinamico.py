
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SVMSMOTE
from imblearn.under_sampling import NearMiss, ClusterCentroids
from sklearn.metrics import accuracy_score, f1_score
import decodIRT_OtML as dIRT_OtML
import decodIRT_MLtIRT as dIRT_MLtIRT
import sys
from IPython.display import display
from openpyxl import Workbook

list_tecnica = ['adasyn','smote','smoten','svmsmote','sem_nada']
numseed =[10, 20, 30]

for tecnica in list_tecnica:
      for seed in numseed:
            data = pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_",seed,"/ilpd_over_",tecnica,"_",seed,"/acc_f1_score.csv")
            print('######################### data ###########################')
            print(data)
            
            
