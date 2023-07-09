
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

#LENDO AS TABELAS COM AS INFORMAÇÕES DE ACURÁCIA E F1 SCORE
data_10= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_30/credit-approval/acc_f1_score.csv")



#TRANSORMANDO AS TABELAS EM DATAFRAME
df10 = pd.DataFrame(data_10)


#MOSTRANDO O DATAFRAME
display(df10)





#MOSTRANDO A MÉDIA DAS ACURÁCIAS
print('A média acc_score é: ',df10["acc_score"].mean())
print('A média f1_score é: ',df10["f1_score"].mean())








#wb = Workbook()

#data.to_excel('ilpd_smote_seed20.xlsx')