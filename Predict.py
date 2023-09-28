#prediction
import os
import pandas as pd
import numpy as np
import csv
import glob
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def process(path,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14):
    data=pd.read_csv("demotest.csv")
    #data=data.drop(['education'],axis=1,inplace=True)
    X_train=data.iloc[1:data.shape[0],0:14]
    y_train=data.iloc[1:data.shape[0],14:15]
    l=[]
    #l.append("eswar")
    l.append(a1)
    l.append(a2)
    l.append(a3)
    l.append(a4)
    l.append(a5)
    l.append(a6)
    l.append(a7)
    l.append(a8)
    l.append(a9)
    l.append(a10)
    l.append(a11)
    l.append(a12)
    l.append(a13)
    l.append(a14)
    #l.append(a11)
    
    
    X_test =pd.DataFrame([l])
    print("Testing data",X_test)
     
    model2=RandomForestClassifier()
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    print("predicted")
    print(y_pred)
    result=""
    if y_pred[0]==1:
        result="Coronary Artery Disease"
    elif y_pred[0]==2:
        result="Heart Arrhythmias"
        
    elif y_pred[0]==3:
        result="Heart Valve Disease"
    elif y_pred[0]==4:
        result="Pericardial Disease"
    else:
        result="No Disease"
    return result
##process("finaldataset.csv",0,46,1,23,0,0,0,1,285,130,84,23.1,85,130)# Heart Valve Disease
##process("finaldataset.csv",0,61,1,30,0,0,1,0,225,150,95,28.58,65,103)# Coronary Artery Disease
##process("finaldataset.csv",1,48,1,20,0,0,0,0,245,127.5,80,25.34,75,70) #Heart Arrhythmias
##process("dataset.csv",1,43,1,30,0,0,1,0,225,162,107,23.61,93,88)# Pericardial Disease


