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


sd = 30
dataset = 'climate-model-simulation-crashes'
tecnica = 'svmsmote'
#LENDO AS TABELAS COM AS INFORMAÇÕES DE ACURÁCIA E F1 SCORE
data_10= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_10/acc_f1_score.csv")
data_20= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_20/acc_f1_score.csv")
data_30= pd.read_csv("/home/fabricio/Documentos/dadosirt/dados_irt/seed_"+str(sd)+"/"+dataset+"_over_"+tecnica+"_30/acc_f1_score.csv")



#TRANSORMANDO AS TABELAS EM DATAFRAME
df10 = pd.DataFrame(data_10)
df20 = pd.DataFrame(data_20)
df30 = pd.DataFrame(data_30)

#MOSTRANDO O DATAFRAME

seedscc = [df10, df20, df30]


#MOSTRANDO A COLUNA DE ACURACIAS


#MOSTRANDO A MÉDIA DAS ACURÁCIAS
for seeds in seedscc:
            
                    print('#################################')
                    print(seeds.iloc[5])
                    print('##############################')
                    print('')
                    #seeds.to_excel('ilpd_adasyn_'+str(nseed)+'.xlsx')



            


           
            


    
        
    
#print('A média geral é acc_score é: ', medias["acc_score"].mean())


